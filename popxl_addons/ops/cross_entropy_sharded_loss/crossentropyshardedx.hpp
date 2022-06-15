// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CROSSENTROPYSHARDEDX_HPP
#define GUARD_NEURALNET_CROSSENTROPYSHARDEDX_HPP

#include <vector>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class CrossEntropyShardedOpx : public PopOpx {
public:
  CrossEntropyShardedOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const;

  poplar::Tensor negLogSoftmax(poplar::Graph &graph,
                               poplar::program::Sequence &prog,
                               const poplar::Tensor &logits) const;

  poplar::Tensor takeTrue(poplar::Graph &graph,
                          poplar::program::Sequence &prog,
                          poplar::Tensor &negLogSoftmax,
                          poplar::Tensor &indicesConcat) const;

protected:
  int32_t groupSize;
  float availableMemoryProportion;
};

class CrossEntropyShardedGradOpx : public PopOpx {
public:
  CrossEntropyShardedGradOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const;

protected:
  int logSoftmaxIndex;
  int logitsIndex;
  int labelsIndex;
  float availableMemoryProportion;
};

} // namespace popx
} // namespace popart

#endif