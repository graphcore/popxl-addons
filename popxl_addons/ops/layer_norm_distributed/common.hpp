// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_LAYERNORMDISTRIBUTED_OPIDS
#define GUARD_LAYERNORMDISTRIBUTED_OPIDS

#include <popart/attributes.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/operatoridentifier.hpp>

using InMapType = std::map<popart::InIndex, popart::TensorId>;
using OutMapType = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex = int;

namespace popart {

#define CUSTOM_OP_DOMAIN "popxl.addons.ops"

const popart::OperatorIdentifier LayerNormDistributed = OperatorIdentifier{
    CUSTOM_OP_DOMAIN,
    "LayerNormDistributed",
    1,   // Op version
    {3}, // number of inputs
    3    // number of outputs
};

const popart::OperatorIdentifier LayerNormDistributedGrad = OperatorIdentifier{
    CUSTOM_OP_DOMAIN,
    "LayerNormDistributedGrad",
    1,   // Op version
    {3}, // number of inputs
    3    // number of outputs
};

} // namespace popart

#endif
