// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <utility>
#include <poplar/Graph.hpp>
#include <poputil/DebugInfo.hpp>

using namespace poplar;

Tensor allReduceStrided(Graph &graph,
                        const Tensor &data,
                        program::Sequence &prog,
                        uint32_t stride,
                        uint32_t size,
                        const DebugContext &debugContext = {},
                        const OptionFlags &options       = {});

Tensor maskedAllGather(Graph &graph,
                       const Tensor &data,
                       program::Sequence &prog,
                       uint32_t stride,
                       uint32_t group);
Tensor maskedReduceScatter(Graph &graph,
                           const Tensor &data,
                           program::Sequence &prog,
                           uint32_t stride,
                           uint32_t group);
Tensor ringAllGather(Graph &graph,
                     const Tensor &data,
                     program::Sequence &prog,
                     uint32_t stride,
                     uint32_t group);
Tensor ringReduceScatter(Graph &graph,
                         const Tensor &data,
                         program::Sequence &prog,
                         uint32_t stride,
                         uint32_t group);

template <typename T>
void initializeTensor(poplar::Graph &graph,
                      poplar::program::Sequence &program,
                      poplar::Tensor &t,
                      T value) {
  poplar::Tensor v =
      graph.addConstant(t.elementType(), {1}, poplar::ArrayRef<T>({value}));
  graph.setTileMapping(v, 1);
  program.add(poplar::program::Copy(
      v.broadcast(t.numElements(), 0).reshape(t.shape()), t));
}
