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
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

Tensor maskedAllGatherCopy(Graph &graph,
                           const Tensor &data,
                           program::Sequence &prog,
                           uint32_t stride,
                           uint32_t group) {
  gcl::CommGroup cgroup{gcl::CommGroupType::CONSECUTIVE, group * stride};
  Tensor temp = gcl::allGatherCrossReplica(graph, data, prog, cgroup);
  std::vector<size_t> shape = data.shape();
  shape.insert(shape.begin(), group);
  Tensor res = graph.addVariable(data.elementType(), shape, {"gather"});
  poputil::mapTensorLinearly(graph, res, 1, 1); // TODO: better mapping ?

  Tensor groupID = graph.addVariable(poplar::UNSIGNED_INT, {}, {"groupID"});
  graph.setTileMapping(groupID, 1); // TODO: find better tile ?
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  popops::mapInPlace(graph,
                     popops::expr::_2 % stride,
                     {groupID, replicaIndex},
                     prog,
                     {"computeGroupID"});
  std::vector<std::pair<std::int32_t, program::Program>> switchBody;
  for (int32_t i = 0; i < int32_t(stride); ++i) {
    program::Sequence copies;
    for (uint32_t j = 0; j < group; ++j) {
      copies.add(program::Copy(temp[i + j * stride], res[j]));
    }
    switchBody.push_back({i, copies});
  }
  prog.add(program::Switch(groupID, switchBody, {"switchCopies"}));
  return res;
}

Tensor maskedAllGather(Graph &graph,
                       const Tensor &data,
                       program::Sequence &prog,
                       uint32_t stride,
                       uint32_t group) {
  gcl::CommGroup cgroup{gcl::CommGroupType::CONSECUTIVE, group * stride};
  Tensor temp = gcl::allGatherCrossReplica(graph, data, prog, cgroup);
  std::vector<size_t> shape = data.shape();
  shape.insert(shape.begin(), group);
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  std::vector<uint32_t> strides, zeroes(group, 0);
  for (uint32_t i = 0; i < group; ++i) {
    strides.push_back(i * stride);
  }
  ArrayRef<uint32_t> ref{strides};
  Tensor stridesT =
      graph.addConstant(poplar::UNSIGNED_INT, {group, 1}, ref, {"strides"});
  graph.setTileMapping(stridesT, 1);
  Tensor offsets_1 = graph.addConstant(
      poplar::UNSIGNED_INT, {group, 1}, ArrayRef<uint32_t>{zeroes}, {"zeroes"});
  graph.setTileMapping(offsets_1, 1);
  Tensor offsets_0 = popops::map(graph,
                                 popops::expr::_1 + (popops::expr::_2 % stride),
                                 {stridesT, replicaIndex},
                                 prog,
                                 {"offsets"});
  Tensor offsets   = poplar::concat({offsets_0, offsets_1}, 1);
  Tensor res =
      popops::multiSlice(graph,
                         temp.reshape({group * stride, data.numElements()}),
                         offsets,
                         {0, 1},
                         {1, data.numElements()},
                         prog,
                         {},
                         {},
                         {"multiCopy"});
  return res.reshape(shape);
}

Tensor maskedAllGatherConcat(Graph &graph,
                             const Tensor &data,
                             program::Sequence &prog,
                             uint32_t stride,
                             uint32_t group) {
  gcl::CommGroup cgroup{gcl::CommGroupType::CONSECUTIVE, group * stride};
  Tensor temp = gcl::allGatherCrossReplica(graph, data, prog, cgroup);
  std::vector<size_t> shape = data.shape();
  shape.insert(shape.begin(), group);
  Tensor res = graph.addVariable(data.elementType(), shape, {"gather"});
  poputil::mapTensorLinearly(graph, res, 1, 1); // TODO: better mapping ?

  Tensor groupID = graph.addVariable(poplar::UNSIGNED_INT, {}, {"groupID"});
  graph.setTileMapping(groupID, 1); // TODO: find better tile ?
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  popops::mapInPlace(graph,
                     popops::expr::_2 % stride,
                     {groupID, replicaIndex},
                     prog,
                     {"computeGroupID"});
  std::vector<std::pair<std::int32_t, program::Program>> switchBody;
  for (int32_t i = 0; i < int32_t(stride); ++i) {
    std::vector<Tensor> regions;
    for (uint32_t j = 0; j < group; ++j) {
      regions.push_back(temp[i + j * stride]);
    }
    switchBody.push_back(
        {i, program::Copy(poplar::concat(regions).reshape(res.shape()), res)});
  }
  prog.add(program::Switch(groupID, switchBody, {"switchCopies"}));
  return res;
}

Tensor maskedAllReduce(Graph &graph,
                       const Tensor &data,
                       program::Sequence &prog,
                       uint32_t stride,
                       uint32_t group) {
  gcl::CommGroup cgroup{gcl::CommGroupType::CONSECUTIVE, group * stride};
  Tensor temp = gcl::allGatherCrossReplica(graph, data, prog, cgroup);
  std::vector<size_t> shape = data.shape();
  shape.insert(shape.begin(), group);
  Tensor res = graph.addVariable(data.elementType(), shape, {"gather"});
  poputil::mapTensorLinearly(graph, res, 1, 1); // TODO: better mapping ?

  Tensor groupID = graph.addVariable(poplar::UNSIGNED_INT, {}, {"groupID"});
  graph.setTileMapping(groupID, 1); // TODO: find better tile ?
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  popops::mapInPlace(graph,
                     popops::expr::_2 % stride,
                     {groupID, replicaIndex},
                     prog,
                     {"computeGroupID"});
  // prog.add(program::PrintTensor("index", replicaIndex));
  // prog.add(program::PrintTensor("id", groupID));
  std::vector<std::pair<std::int32_t, program::Program>> switchBody;
  for (int32_t i = 0; i < int32_t(stride); ++i) {
    program::Sequence copies;
    for (uint32_t j = 0; j < group; ++j) {
      copies.add(program::Copy(temp[i + j * stride], res[j]));
    }
    switchBody.push_back({i, copies});
  }
  prog.add(program::Switch(groupID, switchBody, {"switchCopies"}));
  popops::ReduceParams params(popops::Operation::ADD);
  poplar::Tensor sum = popops::reduce(graph,
                                      res.reshape({group, data.numElements()}),
                                      data.elementType(),
                                      {0},
                                      params,
                                      prog,
                                      {"sum"});
  return sum.reshape(data.shape());
}

Tensor maskedAllReduceConcat(Graph &graph,
                             const Tensor &data,
                             program::Sequence &prog,
                             uint32_t stride,
                             uint32_t group) {
  gcl::CommGroup cgroup{gcl::CommGroupType::CONSECUTIVE, group * stride};
  Tensor temp = gcl::allGatherCrossReplica(graph, data, prog, cgroup);
  std::vector<size_t> shape = data.shape();
  shape.insert(shape.begin(), group);
  Tensor sum = graph.addVariable(data.elementType(), data.shape(), {"sum"});
  poputil::mapTensorLinearly(graph, sum, 1, 1); // TODO: better mapping ?
  Tensor groupID = graph.addVariable(poplar::UNSIGNED_INT, {}, {"groupID"});
  graph.setTileMapping(groupID, 1); // TODO: find better tile ?
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  popops::mapInPlace(graph,
                     popops::expr::_2 % stride,
                     {groupID, replicaIndex},
                     prog,
                     {"computeGroupID"});
  popops::ReduceParams params(popops::Operation::ADD);
  std::vector<std::pair<std::int32_t, program::Program>> switchBody;
  for (int32_t i = 0; i < int32_t(stride); ++i) {
    std::vector<Tensor> regions;
    program::Sequence reduction;
    for (uint32_t j = 0; j < group; ++j) {
      regions.push_back(temp[i + j * stride]);
    }
    Tensor full = poplar::concat(regions).reshape({group, data.numElements()});
    popops::reduceWithOutput(graph, full, sum, {0}, params, reduction, {"sum"});
    switchBody.push_back({i, reduction});
  }
  prog.add(program::Switch(groupID, switchBody, {"switchCopies"}));
  return sum;
}

Tensor maskedReduceScatter(Graph &graph,
                           const Tensor &data,
                           program::Sequence &prog,
                           uint32_t stride,
                           uint32_t group) {
  assert(data.rank() == 1);
  gcl::CommGroup cgroup{gcl::CommGroupType::CONSECUTIVE, group * stride};
  Tensor temp = gcl::allGatherCrossReplica(graph, data, prog, cgroup);
  std::vector<size_t> shape = data.shape();
  shape.insert(shape.begin(), group);
  Tensor res = graph.addVariable(data.elementType(), shape, {"gather"});
  poputil::mapTensorLinearly(graph, res, 1, 1); // TODO: better mapping ?

  Tensor groupID = graph.addVariable(poplar::UNSIGNED_INT, {}, {"groupID"});
  graph.setTileMapping(groupID, 1); // TODO: find better tile ?
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  popops::mapInPlace(graph,
                     popops::expr::_2 % stride,
                     {groupID, replicaIndex},
                     prog,
                     {"computeGroupID"});
  std::vector<std::pair<std::int32_t, program::Program>> switchBody;
  for (int32_t i = 0; i < int32_t(stride); ++i) {
    program::Sequence copies;
    for (uint32_t j = 0; j < group; ++j) {
      copies.add(program::Copy(temp[i + j * stride], res[j]));
    }
    switchBody.push_back({i, copies});
  }
  prog.add(program::Switch(groupID, switchBody, {"switchCopies"}));
  popops::ReduceParams params(popops::Operation::ADD);
  poplar::Tensor sum = popops::reduce(graph,
                                      res.reshape({group, data.numElements()}),
                                      data.elementType(),
                                      {0},
                                      params,
                                      prog,
                                      {"sum"});

  uint32_t replicaSize = std::ceil(float(data.numElements()) / float(group));
  uint32_t dataSize    = data.numElements();
  Tensor scatter =
      graph.addVariable(data.elementType(), {replicaSize}, {"scatter"});
  poputil::mapTensorLinearly(graph, scatter, 1, 1); // TODO: better mapping ?
  initializeTensor(graph, prog, scatter, 0.0);
  Tensor start =
      popops::map(graph,
                  popops::expr::Min(replicaSize * (popops::expr::_1 / stride),
                                    popops::expr::Const(dataSize)),
                  {replicaIndex},
                  prog,
                  {"startIndex"});
  Tensor end = popops::map(
      graph,
      popops::expr::Min(replicaSize * ((popops::expr::_1 / stride) + 1),
                        popops::expr::Const(dataSize)),
      {replicaIndex},
      prog,
      {"endIndex"});
  Tensor size = popops::map(
      graph, popops::expr::_1 - popops::expr::_2, {end, start}, prog, {"size"});
  std::vector<uint32_t> sizes;
  sizes.push_back(replicaSize);
  if (dataSize < group) {
    sizes.push_back(0);
  } else {
    if ((dataSize % replicaSize) != 0)
      sizes.push_back(dataSize % replicaSize);
  }
  std::vector<std::pair<std::int32_t, program::Program>> switchUpdates;
  for (uint32_t s : sizes) {
    program::Sequence update;
    if (s != 0) {
      popops::dynamicSliceWithOutput(graph,
                                     scatter.slice(0, s),
                                     sum,
                                     start.expand({0}),
                                     {0},
                                     {s},
                                     update,
                                     {"scatterUpdate"});
    }
    switchUpdates.push_back({s, update});
  }
  prog.add(program::Switch(size, switchUpdates, {"switchUpdate"}));

  return scatter;
}

std::map<unsigned int, unsigned int>
createRing(const Graph &graph, uint32_t stride, uint32_t group) {
  unsigned int pod = graph.getTarget().getIpuLinkDomainSize();
  assert(stride * group == pod);
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

void allReduceInit(const Tensor &data,
                   program::Sequence &prog,
                   const Tensor &accum,
                   const Tensor &src,
                   const Tensor &src2) {
  prog.add(program::Copy(data, src));
  prog.add(program::Copy(data, src2));
  prog.add(program::Copy(data, accum));
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
                     uint32_t group) {
  std::map<unsigned int, unsigned int> ring = createRing(graph, stride, group),
                                       revRing = reverseMap(ring);

  std::vector<size_t> shape = data.shape();
  Tensor accum = graph.addVariable(data.elementType(), shape, {"accum"});
  poputil::mapTensorLinearly(graph, accum, 1, 1); // TODO: better mapping ?
  Tensor src = graph.addVariable(data.elementType(), data.shape(), {"source"});
  poputil::mapTensorLinearly(graph, src, 1, 1); // TODO: better mapping ?
  Tensor dst =
      graph.addVariable(data.elementType(), data.shape(), {"destination"});
  poputil::mapTensorLinearly(graph, dst, 1, 1); // TODO: better mapping ?
  Tensor src2 = graph.addVariable(data.elementType(), data.shape(), {"source"});
  poputil::mapTensorLinearly(graph, src2, 1, 1); // TODO: better mapping ?
  Tensor dst2 =
      graph.addVariable(data.elementType(), data.shape(), {"destination"});
  poputil::mapTensorLinearly(graph, dst2, 1, 1); // TODO: better mapping ?

  Tensor one = graph.addConstant(poplar::UNSIGNED_INT, {1}, 1, {"one"});
  graph.setTileMapping(one, 1); // TODO: find better tile ?
  Tensor step = graph.addVariable(poplar::UNSIGNED_INT, {}, {"index"});
  graph.setTileMapping(step, 1); // TODO: find better tile ?
  prog.add(program::Copy(one, step));
  allReduceInit(data, prog, accum, src, src2);
  program::Sequence loop = allReduceLoop(
      graph, accum, src, src2, dst, dst2, step, ring, revRing, stride, true);
  prog.add(program::Repeat((ring.size() / 4) - 1, loop, {"loop"}));
  program::Sequence endLoop = allReduceLoop(
      graph, accum, src, src2, dst, dst2, step, ring, revRing, stride, false);
  prog.add(endLoop);
  return accum;
}

Tensor ringReduceScatter(Graph &graph,
                         const Tensor &data,
                         program::Sequence &prog,
                         uint32_t stride,
                         uint32_t group) {
  Tensor sum          = ringAllReduce(graph, data, prog, stride, group);
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  uint32_t replicaSize = std::ceil(float(data.numElements()) / float(group));
  uint32_t dataSize    = data.numElements();
  Tensor scatter =
      graph.addVariable(data.elementType(), {replicaSize}, {"scatter"});
  poputil::mapTensorLinearly(graph, scatter, 1, 1); // TODO: better mapping ?
  initializeTensor(graph, prog, scatter, 0.0);
  Tensor start =
      popops::map(graph,
                  popops::expr::Min(replicaSize * (popops::expr::_1 / stride),
                                    popops::expr::Const(dataSize)),
                  {replicaIndex},
                  prog,
                  {"startIndex"});
  Tensor end = popops::map(
      graph,
      popops::expr::Min(replicaSize * ((popops::expr::_1 / stride) + 1),
                        popops::expr::Const(dataSize)),
      {replicaIndex},
      prog,
      {"endIndex"});
  Tensor size = popops::map(
      graph, popops::expr::_1 - popops::expr::_2, {end, start}, prog, {"size"});
  std::vector<uint32_t> sizes;
  sizes.push_back(replicaSize);
  if (dataSize < group) {
    sizes.push_back(0);
  } else {
    if ((dataSize % replicaSize) != 0)
      sizes.push_back(dataSize % replicaSize);
  }
  std::vector<std::pair<std::int32_t, program::Program>> switchUpdates;
  for (uint32_t s : sizes) {
    program::Sequence update;
    if (s != 0) {
      popops::dynamicSliceWithOutput(graph,
                                     scatter.slice(0, s),
                                     sum,
                                     start.expand({0}),
                                     {0},
                                     {s},
                                     update,
                                     {"scatterUpdate"});
    }
    switchUpdates.push_back({s, update});
  }
  prog.add(program::Switch(size, switchUpdates, {"switchUpdate"}));

  return scatter;
}

void allGatherInit(Graph &graph,
                   const Tensor &data,
                   program::Sequence &prog,
                   const Tensor &res,
                   const Tensor &index,
                   const Tensor &src,
                   const Tensor &src2) {
  prog.add(program::Copy(data, src));
  prog.add(program::Copy(data, src2));
  popops::dynamicUpdate(graph,
                        res,
                        data.expand({0}),
                        index.expand({0}),
                        {0},
                        {1},
                        prog,
                        {"initialCopy"});
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
                     uint32_t group) {
  std::map<unsigned int, unsigned int> ring = createRing(graph, stride, group),
                                       revRing = reverseMap(ring);

  std::vector<size_t> shape = data.shape();
  shape.insert(shape.begin(), group);
  Tensor res = graph.addVariable(data.elementType(), shape, {"gather"});
  poputil::mapTensorLinearly(graph, res, 1, 1); // TODO: better mapping ?
  Tensor src = graph.addVariable(data.elementType(), data.shape(), {"source"});
  poputil::mapTensorLinearly(graph, src, 1, 1); // TODO: better mapping ?
  Tensor dst =
      graph.addVariable(data.elementType(), data.shape(), {"destination"});
  poputil::mapTensorLinearly(graph, dst, 1, 1); // TODO: better mapping ?

  Tensor src2 = graph.addVariable(data.elementType(), data.shape(), {"source"});
  poputil::mapTensorLinearly(graph, src2, 1, 1); // TODO: better mapping ?
  Tensor dst2 =
      graph.addVariable(data.elementType(), data.shape(), {"destination"});
  poputil::mapTensorLinearly(graph, dst2, 1, 1); // TODO: better mapping ?

  Tensor index1 = graph.addVariable(poplar::UNSIGNED_INT, {}, {"index"});
  graph.setTileMapping(index1, 1); // TODO: find better tile ?
  Tensor index2 = graph.addVariable(poplar::UNSIGNED_INT, {}, {"index"});
  graph.setTileMapping(index2, 1); // TODO: find better tile ?
  Tensor replicaIndex = graph.addReplicationIndexConstant({"replicaIndex"});
  graph.setTileMapping(replicaIndex, 1);
  popops::mapInPlace(graph,
                     (popops::expr::_2 / stride),
                     {index1, replicaIndex},
                     prog,
                     {"computeIndex"});
  prog.add(program::Copy(index1, index2));
  allGatherInit(graph, data, prog, res, index1, src, src2);
  Tensor one = graph.addConstant(poplar::UNSIGNED_INT, {1}, 1, {"one"});
  graph.setTileMapping(one, 1); // TODO: find better tile ?
  Tensor step = graph.addVariable(poplar::UNSIGNED_INT, {}, {"step"});
  graph.setTileMapping(step, 1); // TODO: find better tile ?
  prog.add(program::Copy(one, step));
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
                                         group,
                                         true);
  prog.add(program::Repeat((ring.size() / 4) - 1, loop, {"loop"}));
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
                                            group,
                                            false);
  prog.add(endLoop);
  return res;
}
