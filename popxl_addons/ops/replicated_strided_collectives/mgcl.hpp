// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#pragma once

#include <gcl/Collectives.hpp>
#include <poplar/Graph.hpp>
#include <poputil/DebugInfo.hpp>
#include <utility>

using namespace poplar;

Tensor allReduceStrided(Graph &graph, const Tensor &data,
                        program::Sequence &prog, gcl::CollectiveOperator op,
                        uint32_t stride, uint32_t size,
                        const DebugContext &debugContext = {},
                        const OptionFlags &options = {});

Tensor reduceScatterStrided(Graph &graph, const Tensor &data,
                            program::Sequence &prog, gcl::CollectiveOperator op,
                            uint32_t stride, uint32_t size,
                            const DebugContext &debugContext = {},
                            const OptionFlags &options = {});

Tensor allGatherStrided(Graph &graph, const Tensor &data,
                        program::Sequence &prog, uint32_t stride, uint32_t size,
                        const DebugContext &debugContext = {},
                        const OptionFlags &options = {});
