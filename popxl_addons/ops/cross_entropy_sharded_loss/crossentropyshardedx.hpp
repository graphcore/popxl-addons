// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CROSSENTROPYSHARDEDX_HPP
#define GUARD_NEURALNET_CROSSENTROPYSHARDEDX_HPP

#include <popart/names.hpp>
#include <popart/popx/opx.hpp>
#include <popart/replicagrouping.hpp>
#include <vector>

namespace popart {
namespace popx {

class CrossEntropyShardedOpx : public Opx {
public:
  CrossEntropyShardedOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const;

  poplar::Tensor negLogSoftmax(poplar::Graph &graph,
                               poplar::program::Sequence &prog,
                               const poplar::Tensor &logits) const;

  poplar::Tensor takeTrue(poplar::Graph &graph, poplar::program::Sequence &prog,
                          poplar::Tensor &negLogSoftmax,
                          poplar::Tensor &indicesConcat) const;

protected:
  float availableMemoryProportion;
};

class CrossEntropyShardedGradOpx : public Opx {
public:
  CrossEntropyShardedGradOpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const;

protected:
  int logSoftmaxIndex;
  int logitsIndex;
  int labelsIndex;
  float availableMemoryProportion;
};

} // namespace popx
} // namespace popart

#endif