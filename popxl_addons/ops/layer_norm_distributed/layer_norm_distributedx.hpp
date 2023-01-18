// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_OP_LAYERNORMDISTRIBUTEDX_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_OP_LAYERNORMDISTRIBUTEDX_HPP_

#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/op/normx.hpp>

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;

// class LayerNormDistributedOpx : public NormOpx {
class LayerNormDistributedOpx : public CollectivesBaseOpx {
public:
  LayerNormDistributedOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
};

// class LayerNormDistributedGradOpx : public NormOpx {
class LayerNormDistributedGradOpx : public CollectivesBaseOpx {
public:
  LayerNormDistributedGradOpx(Op *, Devicex *);
  void grow(poplar::program::Sequence &) const final;

private:
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_POPX_OP_LAYERNORMDISTRIBUTEDX_HPP_
