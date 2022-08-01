// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ROTARYPOSEMBEDX_HPP
#define GUARD_NEURALNET_ROTARYPOSEMBEDX_HPP

#include <vector>
#include <popart/names.hpp>
#include <popart/popx/popopx.hpp>

namespace popart {
namespace popx {

class RotaryPosEmbedOpx : public PopOpx {
public:
  RotaryPosEmbedOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const;
};

class RotaryPosEmbedGradOpx : public PopOpx {
public:
  RotaryPosEmbedGradOpx(Op *, Devicex *);

  void grow(snap::program::Sequence &) const;
};

} // namespace popx
} // namespace popart

#endif