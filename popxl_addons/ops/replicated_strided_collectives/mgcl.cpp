// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include "mgcl.hpp"
#include <cmath>
#include <gcl/Collectives.hpp>
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

gcl::CommGroup consecutiveGroup(Graph &graph, uint32_t size) {
  // Handle GCL's preference of ALL
  if (graph.getReplicationFactor() == size)
    return {gcl::CommGroupType::ALL, 0};
  return {gcl::CommGroupType::CONSECUTIVE, size};
}

Tensor maskedAllGather(Graph &graph,
                       const Tensor &data,
                       program::Sequence &prog,
                       uint32_t stride,
                       uint32_t size) {
  // Gather all Replica's data
  Tensor temp = gcl::allGatherCrossReplica(
      graph, data, prog, consecutiveGroup(graph, size * stride));

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
      graph, data, prog, consecutiveGroup(graph, size * stride));

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

std::map<unsigned int, unsigned int>
createRing(const Graph &graph, uint32_t stride, uint32_t size) {
  unsigned int pod = graph.getTarget().getIpuLinkDomainSize();
  assert(stride * size == pod);
  std::map<unsigned int, unsigned int> ring;
  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t j = 0; j < pod; j += 2) {
      uint32_t index = i + j;
      ring[index]    = (index + 2) % pod;
    }
  }
  return ring;
}

std::map<unsigned int, unsigned int>
reverseMap(const std::map<unsigned int, unsigned int> &ring) {
  std::map<unsigned int, unsigned int> res;
  for (const auto &p : ring) {
    res[p.second] = p.first;
  }
  return res;
}

program::Sequence
allReduceLoop(Graph &graph,
              const Tensor &accum,
              const Tensor &src,
              const Tensor &src2,
              const Tensor &dest,
              const Tensor &dest2,
              const Tensor &step,
              const std::map<unsigned int, unsigned int> &ring,
              const std::map<unsigned int, unsigned int> &revRing,
              uint32_t stride,
              bool full) {
  program::Sequence loop;
  loop.add(program::CrossReplicaCopy(src, dest, ring, {"CrossReplica1"}));
  if (full)
    loop.add(
        program::CrossReplicaCopy(src2, dest2, revRing, {"CrossReplica2"}));
  uint32_t halfStride = stride / 2;
  Tensor strideT      = popops::map(
      graph, popops::expr::_1 % halfStride, {step}, loop, {"strideIndex"});
  std::vector<std::pair<std::int32_t, program::Program>> switchBody;
  program::Sequence sum;
  if (full)
    popops::mapInPlace(graph,
                       popops::expr::_1 + popops::expr::_2 + popops::expr::_3,
                       {accum, dest, dest2},
                       sum,
                       {"sum2"});
  else
    popops::addInPlace(graph, accum, dest, sum, {"sum"});

  switchBody.push_back({0, sum});
  loop.add(program::Switch(strideT, switchBody, {"switchSum"}));
  loop.add(program::Copy(dest, src));
  if (full)
    loop.add(program::Copy(dest2, src2));
  popops::addInPlace(graph, step, 1U, loop, {"incrementStep"});
  return loop;
}

Tensor ringAllReduce(Graph &graph,
                     const Tensor &data,
                     program::Sequence &prog,
                     uint32_t stride,
                     uint32_t size) {
  auto ring    = createRing(graph, stride, size);
  auto revRing = reverseMap(ring);

  // Create Buffers
  Tensor accum = graph.clone(data, {"accum"});
  Tensor src   = graph.clone(data, {"source"});
  Tensor src2  = graph.clone(data, {"source2"});
  Tensor dst   = graph.clone(data, {"destination"});
  Tensor dst2  = graph.clone(data, {"destination2"});
  // Initialise
  prog.add(program::Copy(data, accum));
  prog.add(program::Copy(data, src));
  prog.add(program::Copy(data, src2));
  prog.add(program::WriteUndef(dst));
  prog.add(program::WriteUndef(dst2));

  // Create counters
  Tensor one = graph.addConstant(poplar::UNSIGNED_INT, {1}, 1, {"one"});
  graph.setTileMapping(one, 1);
  Tensor step = graph.addVariable(poplar::UNSIGNED_INT, {}, {"index"});
  graph.setTileMapping(step, 1);
  prog.add(program::Copy(one, step));

  // Loop
  program::Sequence loop = allReduceLoop(
      graph, accum, src, src2, dst, dst2, step, ring, revRing, stride, true);
  prog.add(program::Repeat((ring.size() / 4) - 1, loop, {"loop"}));
  // Final Step
  program::Sequence endLoop = allReduceLoop(
      graph, accum, src, src2, dst, dst2, step, ring, revRing, stride, false);
  prog.add(endLoop);
  return accum;
}

Tensor ringReduceScatter(Graph &graph,
                         const Tensor &data,
                         program::Sequence &prog,
                         uint32_t stride,
                         uint32_t size) {
  assert(data.rank() == 1);

  Tensor reduced_data = ringAllReduce(graph, data, prog, stride, size);
  Tensor scattered =
      reduceScatterSlice(graph, reduced_data, prog, stride, size);

  return scattered;
}

program::Sequence
allGatherLoop(Graph &graph,
              const Tensor &res,
              const Tensor &src,
              const Tensor &src2,
              const Tensor &dest,
              const Tensor &dest2,
              const Tensor &step,
              const Tensor &index1,
              const Tensor &index2,
              const std::map<unsigned int, unsigned int> &ring,
              const std::map<unsigned int, unsigned int> &revRing,
              uint32_t stride,
              uint32_t group,
              bool full) {
  program::Sequence loop;
  loop.add(program::CrossReplicaCopy(src, dest, ring, {"CrossReplica1"}));
  if (full)
    loop.add(
        program::CrossReplicaCopy(src2, dest2, revRing, {"CrossReplica2"}));
  uint32_t halfStride = stride / 2;
  Tensor strideT      = popops::map(
      graph, popops::expr::_1 % halfStride, {step}, loop, {"strideIndex"});
  std::vector<std::pair<std::int32_t, program::Program>> switchBody;
  program::Sequence copies;
  if (full) {
    popops::mapInPlace(graph,
                       (popops::expr::_1 + 1) % group,
                       {index2},
                       copies,
                       {"decrementIndex2"});
    popops::dynamicUpdate(graph,
                          res,
                          dest2.expand({0}),
                          index2.expand({0}),
                          {0},
                          {1},
                          copies,
                          {"copyBackward"});
  }
  popops::mapInPlace(graph,
                     (popops::expr::_1 - 1) % group,
                     {index1},
                     copies,
                     {"indcrementIndex1"});
  popops::dynamicUpdate(graph,
                        res,
                        dest.expand({0}),
                        index1.expand({0}),
                        {0},
                        {1},
                        copies,
                        {"copyForward"});

  switchBody.push_back({0, copies});
  loop.add(program::Switch(strideT, switchBody, {"switchCopies"}));
  loop.add(program::Copy(dest, src));
  if (full)
    loop.add(program::Copy(dest2, src2));
  popops::addInPlace(graph, step, 1U, loop, {"incrementStep"});
  return loop;
}

Tensor ringAllGather(Graph &graph,
                     const Tensor &data,
                     program::Sequence &prog,
                     uint32_t stride,
                     uint32_t size) {
  auto ring    = createRing(graph, stride, size);
  auto revRing = reverseMap(ring);

  // Calculate index
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  Tensor index1 = popops::map(graph,
                              (popops::expr::_1 / stride),
                              {replicaIndex},
                              prog,
                              {"computeIndex"});
  Tensor index2 = poputil::duplicate(graph, index1, prog);

  // Create buffers
  Tensor res  = graph.cloneN(data, size);
  Tensor src  = graph.clone(data);
  Tensor src2 = graph.clone(data);
  Tensor dst  = graph.clone(data);
  Tensor dst2 = graph.clone(data);

  // Initialise
  popops::dynamicUpdate(graph,
                        res,
                        data.expand({0}),
                        index1.expand({0}),
                        {0},
                        {1},
                        prog,
                        {"initialCopy"});
  prog.add(program::Copy(data, src));
  prog.add(program::Copy(data, src2));
  prog.add(program::WriteUndef(dst));
  prog.add(program::WriteUndef(dst2));

  // Setup counters
  Tensor one = graph.addConstant(poplar::UNSIGNED_INT, {1}, 1, {"one"});
  graph.setTileMapping(one, 1);
  Tensor step = graph.addVariable(poplar::UNSIGNED_INT, {}, {"step"});
  graph.setTileMapping(step, 1);
  prog.add(program::Copy(one, step));

  // Loop
  program::Sequence loop = allGatherLoop(graph,
                                         res,
                                         src,
                                         src2,
                                         dst,
                                         dst2,
                                         step,
                                         index1,
                                         index2,
                                         ring,
                                         revRing,
                                         stride,
                                         size,
                                         true);
  prog.add(program::Repeat((ring.size() / 4) - 1, loop, {"loop"}));

  // Final Step
  program::Sequence endLoop = allGatherLoop(graph,
                                            res,
                                            src,
                                            src2,
                                            dst,
                                            dst2,
                                            step,
                                            index1,
                                            index2,
                                            ring,
                                            revRing,
                                            stride,
                                            size,
                                            false);
  prog.add(endLoop);
  return res;
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
  if (stride == 1) {
    return gcl::allReduceCrossReplica(graph,
                                      data,
                                      op,
                                      prog,
                                      consecutiveGroup(graph, size * stride),
                                      debugContext,
                                      options);
  } else {
    if (stride == 64) {
      return gcl::allReduceCrossReplica(
          graph,
          data,
          op,
          prog,
          gcl::CommGroup(gcl::CommGroupType::ORTHOGONAL, stride),
          debugContext,
          options);
    } else {
      Tensor out;
      if (stride * size == 64) {
        out = ringAllReduce(graph, data, prog, stride, size);
      } else {
        out = maskedAllReduce(graph, data, prog, stride, size);
      }
      if (op == gcl::CollectiveOperator::MEAN) {
        popops::mulInPlace(
            graph, out, 1.f / static_cast<float>(size), prog, debugContext);
      }
      return out;
    }
  }
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
  if (stride == 1) {
    return gcl::reduceScatterCrossReplica(
        graph,
        data,
        op,
        prog,
        consecutiveGroup(graph, size * stride),
        debugContext,
        options);
  } else {
    if (stride == 64) {
      return gcl::reduceScatterCrossReplica(
          graph,
          data,
          op,
          prog,
          gcl::CommGroup(gcl::CommGroupType::ORTHOGONAL, stride),
          debugContext,
          options);
    } else {
      Tensor out;
      if (stride * size == 64) {
        out = ringReduceScatter(graph, data, prog, stride, size);
      } else {
        out = maskedReduceScatter(graph, data, prog, stride, size);
      }
      if (op == gcl::CollectiveOperator::MEAN) {
        popops::mulInPlace(
            graph, out, 1.f / static_cast<float>(size), prog, debugContext);
      }
      return out;
    }
  }
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
  if (stride == 1) {
    return gcl::allGatherCrossReplica(graph,
                                      data,
                                      prog,
                                      consecutiveGroup(graph, size * stride),
                                      debugContext,
                                      options);
  } else {
    if (stride == 64) {
      return gcl::allGatherCrossReplica(
          graph,
          data,
          prog,
          gcl::CommGroup(gcl::CommGroupType::ORTHOGONAL, stride),
          debugContext,
          options);
    } else {
      if (stride * size == 64) {
        return ringAllGather(graph, data, prog, stride, size);
      } else {
        return maskedAllGather(graph, data, prog, stride, size);
      }
    }
  }
}

// -------- All Gather --------