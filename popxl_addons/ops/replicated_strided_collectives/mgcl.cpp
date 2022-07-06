// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "mgcl.hpp"
#include <cmath>
#include <gcl/Collectives.hpp>
#include <poplar/Target.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <popnn/Loss.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

namespace {

bool ringSupported(Graph &graph, uint32_t stride, uint32_t size) {
  auto &target = graph.getTarget();
  return stride * size == target.getIpuLinkDomainSize() &&
         target.getIpuLinkTopology() == IpuLinkTopology::Torus;
}

gcl::CommGroup consecutiveGroup(Graph &graph, uint32_t stride, uint32_t size) {
  // Handle GCL's preference of ALL
  if (graph.getReplicationFactor() == size)
    return {gcl::CommGroupType::ALL, 0};
  return {gcl::CommGroupType::CONSECUTIVE, size, stride};
}

Tensor maskedAllGather(Graph &graph,
                       const Tensor &data,
                       program::Sequence &prog,
                       uint32_t stride,
                       uint32_t size) {
  // Gather all Replica's data
  Tensor temp = gcl::allGatherCrossReplica(
      graph, data, prog, consecutiveGroup(graph, 1, size * stride));

  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);

  // Calculate multislice offsets
  std::vector<uint32_t> strides, zeroes(size, 0);
  for (uint32_t i = 0; i < size; ++i) {
    strides.push_back(i * stride);
  }
  ArrayRef<uint32_t> ref{strides};
  Tensor stridesT =
      graph.addConstant(poplar::UNSIGNED_INT, {size, 1}, ref, {"strides"});
  graph.setTileMapping(stridesT, 1);
  Tensor offsets_1 = graph.addConstant(
      poplar::UNSIGNED_INT, {size, 1}, ArrayRef<uint32_t>{zeroes}, {"zeroes"});
  graph.setTileMapping(offsets_1, 1);
  Tensor offsets = popops::map(graph,
                               popops::expr::_1 + (popops::expr::_2 % stride),
                               {stridesT, replicaIndex},
                               prog,
                               {"offsets"});

  // TODO: Check tile mapping
  Tensor res =
      popops::multiSlice(graph,
                         temp.reshape({size * stride, data.numElements()}),
                         offsets,
                         {0},
                         {1},
                         prog,
                         {},
                         {},
                         {"multiCopy"});

  std::vector<size_t> shape = data.shape();
  shape.insert(shape.begin(), size);
  return res.reshape(shape);
}

Tensor maskedAllReduce(Graph &graph,
                       const Tensor &data,
                       program::Sequence &prog,
                       uint32_t stride,
                       uint32_t size) {
  if (stride == 1) {
    throw std::runtime_error(
        "maskedAllReduce stride == 1 is not implemented. "
        "Use GCL with {gcl::CommGroupType::CONSECUTIVE, size} instead.");
  }

  // Gather all Replica's data
  Tensor temp = gcl::allGatherCrossReplica(
      graph, data, prog, consecutiveGroup(graph, 1, size * stride));

  // Compute GroupId
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  Tensor groupId = popops::map(graph,
                               popops::expr::_1 % stride,
                               {replicaIndex},
                               prog,
                               {"computeGroupId"});

  // Create buffer for group's data
  Tensor group_data = graph.cloneN(data, size);
  prog.add(program::WriteUndef(group_data));

  // Copy Group's data to buffer
  std::vector<std::pair<std::int32_t, program::Program>> switchBody;
  for (int32_t i = 0; i < int32_t(stride); ++i) {
    program::Sequence copies;
    for (uint32_t j = 0; j < size; ++j) {
      copies.add(program::Copy(temp[i + j * stride], group_data[j]));
    }
    switchBody.push_back({i, copies});
  }
  prog.add(program::Switch(groupId, switchBody, {"switchCopies"}));

  // Reduce group_data for final result
  Tensor out = graph.clone(data.flatten());

  popops::reduceWithOutput(graph,
                           group_data.reshape({size, data.numElements()}),
                           out,
                           {0},
                           {popops::Operation::ADD},
                           prog,
                           {"sum"});
  prog.add(program::WriteUndef(group_data));
  return out.reshape(data.shape());
}

Tensor reduceScatterSliceWithoutPadding(Graph &graph,
                                        const Tensor &reduced_data,
                                        program::Sequence &prog,
                                        uint32_t stride,
                                        uint32_t size) {
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  uint32_t dataSize      = reduced_data.numElements();
  uint32_t scatteredSize = std::ceil(float(dataSize) / float(size));

  Tensor scatter =
      graph.clone(reduced_data.slice(0, scatteredSize), {"scatter"});
  popops::zero(graph, scatter, prog, {"zeroScatter"});

  uint32_t outer_stride = stride * size;

  Tensor start = popops::map(
      graph,
      popops::expr::Min(scatteredSize *
                            ((popops::expr::_1 % outer_stride) / stride),
                        popops::expr::Const(dataSize)),
      {replicaIndex},
      prog,
      {"startIndex"});
  Tensor end = popops::map(
      graph,
      popops::expr::Min(scatteredSize *
                            (((popops::expr::_1 % outer_stride) / stride) + 1),
                        popops::expr::Const(dataSize)),
      {replicaIndex},
      prog,
      {"endIndex"});

  Tensor copy_size = popops::map(
      graph, popops::expr::_1 - popops::expr::_2, {end, start}, prog, {"size"});
  std::vector<uint32_t> sizes;
  sizes.push_back(scatteredSize);
  if (dataSize < size) {
    sizes.push_back(0);
  } else {
    if ((dataSize % scatteredSize) != 0)
      sizes.push_back(dataSize % scatteredSize);
  }
  std::vector<std::pair<std::int32_t, program::Program>> switchUpdates;
  for (uint32_t s : sizes) {
    program::Sequence update;
    if (s != 0) {
      popops::dynamicSliceWithOutput(graph,
                                     scatter.slice(0, s),
                                     reduced_data,
                                     start.expand({0}),
                                     {0},
                                     {s},
                                     update,
                                     {"scatterUpdate"});
    }
    switchUpdates.push_back({s, update});
  }
  prog.add(program::Switch(copy_size, switchUpdates, {"switchUpdate"}));

  return scatter;
}

Tensor reduceScatterSlice(Graph &graph,
                          const Tensor &reduced_data,
                          program::Sequence &prog,
                          uint32_t stride,
                          uint32_t size) {
  if ((reduced_data.numElements() % size) != 0) {
    // Do inefficient dynamicSlice
    return reduceScatterSliceWithoutPadding(
        graph, reduced_data, prog, stride, size);
  }
  // More efficient implementation that assumes `reduced_data` can be reshaped
  // to be divisible by size for Tensors that have been reordered/padded for RTS
  // this will always be the case.
  auto scatteredSize   = reduced_data.numElements() / size;
  uint32_t outerStride = stride * size;

  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  Tensor rank = popops::map(graph,
                            (popops::expr::_1 % outerStride) / stride,
                            {replicaIndex},
                            prog,
                            {"startIndex"});

  Tensor dataByRank = reduced_data.reshape({size, scatteredSize});
  Tensor data       = popops::multiSlice(graph,
                                   dataByRank,
                                   rank.reshape({1, 1}),
                                   {0},
                                   {1},
                                   prog,
                                   {},
                                   {},
                                   {"reduceScatterSlice"});

  Tensor scatter = graph.clone(reduced_data.slice(0, scatteredSize));
  prog.add(program::Copy(data, scatter));

  return scatter;
}

Tensor maskedReduceScatter(Graph &graph,
                           const Tensor &data,
                           program::Sequence &prog,
                           uint32_t stride,
                           uint32_t size) {
  assert(data.rank() == 1);

  Tensor reduced_data = maskedAllReduce(graph, data, prog, stride, size);
  Tensor scattered =
      reduceScatterSlice(graph, reduced_data, prog, stride, size);

  return scattered;
}
} // namespace

// -------- All Reduce --------

Tensor allReduceStrided(Graph &graph,
                        const Tensor &data,
                        program::Sequence &prog,
                        gcl::CollectiveOperator op,
                        uint32_t stride,
                        uint32_t size,
                        const DebugContext &debugContext,
                        const OptionFlags &options) {
  if (!(op == gcl::CollectiveOperator::ADD ||
        op == gcl::CollectiveOperator::MEAN)) {
    throw runtime_error(
        "reduceScatterStrided only supports "
        "gcl::CollectiveOperator::ADD and gcl::CollectiveOperator::MEAN");
  }
  if (stride == graph.getTarget().getIpuLinkDomainSize()) {
    return gcl::allReduceCrossReplica(
        graph,
        data,
        op,
        prog,
        gcl::CommGroup(gcl::CommGroupType::ORTHOGONAL, size),
        debugContext,
        options);
  }
  if (stride == 1 || ringSupported(graph, stride, size)) {
    return gcl::allReduceCrossReplica(graph,
                                      data,
                                      op,
                                      prog,
                                      consecutiveGroup(graph, stride, size),
                                      debugContext,
                                      options);
  }
  Tensor out = maskedAllReduce(graph, data, prog, stride, size);
  if (op == gcl::CollectiveOperator::MEAN) {
    popops::mulInPlace(
        graph, out, 1.f / static_cast<float>(size), prog, debugContext);
  }
  return out;
}

// -------- All Reduce --------

// ------ Reduce Scatter ------

Tensor reduceScatterStrided(Graph &graph,
                            const Tensor &data,
                            program::Sequence &prog,
                            gcl::CollectiveOperator op,
                            uint32_t stride,
                            uint32_t size,
                            const DebugContext &debugContext,
                            const OptionFlags &options) {
  if (!(op == gcl::CollectiveOperator::ADD ||
        op == gcl::CollectiveOperator::MEAN)) {
    throw runtime_error(
        "reduceScatterStrided only supports "
        "gcl::CollectiveOperator::ADD and gcl::CollectiveOperator::MEAN");
  }
  if (stride == graph.getTarget().getIpuLinkDomainSize()) {
    return gcl::reduceScatterCrossReplica(
        graph,
        data,
        op,
        prog,
        gcl::CommGroup(gcl::CommGroupType::ORTHOGONAL, size),
        debugContext,
        options);
  }
  if (stride == 1 || ringSupported(graph, stride, size)) {
    return gcl::reduceScatterCrossReplica(graph,
                                          data,
                                          op,
                                          prog,
                                          consecutiveGroup(graph, stride, size),
                                          debugContext,
                                          options);
  }
  Tensor out = maskedReduceScatter(graph, data, prog, stride, size);
  if (op == gcl::CollectiveOperator::MEAN) {
    popops::mulInPlace(
        graph, out, 1.f / static_cast<float>(size), prog, debugContext);
  }
  return out;
}

// ------ Reduce Scatter ------

// -------- All Gather --------

Tensor allGatherStrided(Graph &graph,
                        const Tensor &data,
                        program::Sequence &prog,
                        uint32_t stride,
                        uint32_t size,
                        const DebugContext &debugContext,
                        const OptionFlags &options) {
  if (stride == graph.getTarget().getIpuLinkDomainSize()) {
    return gcl::allGatherCrossReplica(
        graph,
        data,
        prog,
        gcl::CommGroup(gcl::CommGroupType::ORTHOGONAL, size),
        debugContext,
        options);
  }
  if (stride == 1 || ringSupported(graph, stride, size)) {
    return gcl::allGatherCrossReplica(graph,
                                      data,
                                      prog,
                                      consecutiveGroup(graph, stride, size),
                                      debugContext,
                                      options);
  }
  return maskedAllGather(graph, data, prog, stride, size);
}

// -------- All Gather --------