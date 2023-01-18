// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CROSSENTROPYSHARDEDWRX_HPP
#define GUARD_NEURALNET_CROSSENTROPYSHARDEDWRX_HPP

#include <vector>
#include <popart/names.hpp>
#include <popart/popx/opx.hpp>

namespace popart {
namespace popx {

class CrossEntropyShardedWROpx : public Opx {
public:
  CrossEntropyShardedWROpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const;

  std::vector<std::reference_wrapper<poplar::Graph>> getVGraphs() const;

  std::vector<poplar::Tensor> sharded_log_softmax(
      poplar::Graph &graph,
      std::vector<std::reference_wrapper<poplar::Graph>> vGraphs,
      poplar::program::Sequence &prog,
      const std::vector<poplar::Tensor> &logits) const;

  poplar::Tensor
  sharded_take_last(poplar::Graph &graph,
                    std::vector<std::reference_wrapper<poplar::Graph>> vGraphs,
                    poplar::program::Sequence &prog,
                    std::vector<poplar::Tensor> negLogSoftmax,
                    poplar::Tensor indiciesConcat) const;

protected:
  int numShards;
};

class CrossEntropyShardedGradWROpx : public Opx {
public:
  CrossEntropyShardedGradWROpx(Op *, Devicex *);

  void grow(poplar::program::Sequence &) const;

  std::vector<std::reference_wrapper<poplar::Graph>> getVGraphs() const;

protected:
  std::vector<int64_t> ipus;
  int numShards;
  int logSoftmaxStartIndex;
  int logitsStartIndex;
  int labelsStartIndex;
};

} // namespace popx
} // namespace popart

#endif