// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_CROSSENTROPYSHARDEDLOSS_OPIDS
#define GUARD_CROSSENTROPYSHARDEDLOSS_OPIDS

#include <popart/attributes.hpp>
#include <popart/error.hpp>
#include <popart/names.hpp>
#include <popart/operatoridentifier.hpp>

using InMapType = std::map<popart::InIndex, popart::TensorId>;
using OutMapType = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex = int;

namespace popart {

#define CUSTOM_OP_DOMAIN "popxl.addons.ops"

const popart::OperatorIdentifier CrossEntropySharded = OperatorIdentifier{
    CUSTOM_OP_DOMAIN,
    "CrossEntropySharded",
    1,      // Op version
    {1, 1}, // number of inputs
    1       // number of outputs
};

const popart::OperatorIdentifier CrossEntropyShardedGrad = OperatorIdentifier{
    CUSTOM_OP_DOMAIN,
    "CrossEntropyShardedGrad",
    1,      // Op version
    {1, 1}, // number of inputs
    1       // number of outputs
};

} // namespace popart

#endif
