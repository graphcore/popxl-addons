// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "popnn/LayerNorm.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gcl/Collectives.hpp>
#include <iostream>
#include <layer_norm_distributed.hpp>
#include <layer_norm_distributedx.hpp>
#include <popart/ir.hpp>
#include <popart/popx/op/normx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <poplin/Norms.hpp>
#include <popnn/BatchNorm.hpp>
#include <popnn/GroupNorm.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Rearrange.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "popart/graphcoreoperators.hpp"
#include "popart/sessionoptions.hpp"
// #include <popart/op/collectives/collectives.hpp>

namespace popart {
class Op;
namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

namespace poplar {
using Shape = std::vector<std::size_t>;
}

namespace pe = popops::expr;

namespace popart {
namespace popx {

poplin::DistributedNormReduceCallback
GetDistributedNormReduceCallback(const ReplicaGrouping &group_ref,
                                 const std::string &debug_name_ref) {
  const std::string debug_name = debug_name_ref;
  const ReplicaGrouping &group = group_ref;
  return
      [debug_name, group](
          poplar::Graph &graph, const std::vector<poplar::Tensor> &inputs,
          poplar::program::Sequence &prog, unsigned replica_group_size,
          const poplar::DebugContext &debug_context,
          const poplar::OptionFlags &options) -> std::vector<poplar::Tensor> {
        // Use multi-tensor allReduce to reduce them all at the same time even
        // if they have different types.
        return gcl::allReduceCrossReplica(
            graph, inputs, gcl::CollectiveOperator::ADD, prog,
            toGclCommGroup(group), {debug_name}, options);
      };
}

LayerNormDistributedOpx::LayerNormDistributedOpx(Op *op, Devicex *devicex)
    : CollectivesBaseOpx(op, devicex) {
  verifyOp<LayerNormDistributedOp>(op, LayerNormDistributed);
}

void LayerNormDistributedOpx::grow(poplar::program::Sequence &prog) const {

  auto &op = getOp<LayerNormDistributedOp>();

  // Get the attributes
  float epsilon = op.getEpsilon();
  // Check for stable algorithm session option.
  bool stable_algo = op.getIr().getSessionOptions().enableStableNorm;

  // Get the inputs
  auto input = getInTensor(LayerNormDistributedOp::getXInIndex());
  auto scale = getInTensor(LayerNormDistributedOp::getScaleInIndex());
  auto b = getInTensor(LayerNormDistributedOp::getBInIndex());

  // Get hidden dimension size
  auto hiddenSize = input.shape()[1] * op.getReplicaGrouping().getGroupSize();

  // Calculate the mean and the inverse standard deviation
  poplar::Tensor mean;
  poplar::Tensor invStdDev;

  // Swap axis to make batch norm behave like LayerNorm
  auto input_T = input.transpose();
  std::tie(mean, invStdDev) = poplin::distributedNormStatistics(
      graph(), input_T, epsilon, prog, false,
      GetDistributedNormReduceCallback(op.getReplicaGrouping(),
                                       "groupNormAllReduce"),
      hiddenSize, stable_algo, poplar::FLOAT,
      debugContext("groupNormStatistics"));

  // Use original input tensor when applying group normalisation
  auto result =
      popnn::ln::layerNormalise(graph(), input, scale, b, mean, invStdDev, prog,
                                debugContext("layerNorm"))
          .first;

  setOutTensor(LayerNormDistributedOp::getYOutIndex(), result);
  setOutTensor(LayerNormDistributedOp::getMeanOutIndex(), mean);
  setOutTensor(LayerNormDistributedOp::getInvStdDevOutIndex(), invStdDev);
}

LayerNormDistributedGradOpx::LayerNormDistributedGradOpx(Op *op,
                                                         Devicex *devicex)
    : CollectivesBaseOpx(op, devicex) {
  verifyOp<LayerNormDistributedGradOp>(op, LayerNormDistributedGrad);
}

void LayerNormDistributedGradOpx::grow(poplar::program::Sequence &prog) const {

  auto x = getInTensor(LayerNormDistributedGradOp::getXInIndex());
  auto yGrad = getInTensor(LayerNormDistributedGradOp::getYGradInIndex());
  auto scale = getInTensor(LayerNormDistributedGradOp::getScaleInIndex());
  auto mean = getInTensor(LayerNormDistributedGradOp::getMeanInIndex());
  auto invStdDev =
      getInTensor(LayerNormDistributedGradOp::getInvStdDevInIndex());

  auto &op = getOp<LayerNormDistributedGradOp>();

  // Get hidden dimension size
  auto hiddenSize = x.shape()[1] * op.getReplicaGrouping().getGroupSize();

  // Using GroupNorm as it behaves like LayerNorm when num_groups == 1
  poplar::Tensor xWhitened = popnn::gn::groupNormWhiten(
      graph(), x, mean, invStdDev, prog, debugContext("whitenedActs"));

  // Compute the delta for the operand
  // Transpose input to make BatchNorm statistic calculation behave like
  // LayerNorm would.
  poplar::Tensor xGrad = poplin::distributedNormStatisticsGradients(
      graph(), xWhitened.transpose(), yGrad.transpose(), invStdDev, prog,
      GetDistributedNormReduceCallback(op.getReplicaGrouping(),
                                       "groupNormAllReduce"),
      hiddenSize, poplar::FLOAT, debugContext("operandGrad"));

  // Compute the deltas for scaled and offset.
  // Using GroupNorm as it behaves similar to LayerNorm.
  poplar::Tensor scaleGrad;
  poplar::Tensor bGrad;
  std::tie(scaleGrad, bGrad) = popnn::gn::groupNormParamGradients(
      graph(), xWhitened, yGrad, prog, poplar::FLOAT,
      debugContext("scaleOffsetGrads"));

  // Return the result
  setOutTensor(LayerNormDistributedGradOp::getXGradOutIndex(),
               xGrad.transpose());
  setOutTensor(LayerNormDistributedGradOp::getScaleOutIndex(), scaleGrad);
  setOutTensor(LayerNormDistributedGradOp::getBOutIndex(), bGrad);
}

namespace {
OpxCreator<LayerNormDistributedOpx>
    LayerNormDistributedOpxCreator(LayerNormDistributed);
OpxCreator<LayerNormDistributedGradOpx>
    LayerNormDistributedGradOpxCreator(LayerNormDistributedGrad);
} // namespace

} // namespace popx
} // namespace popart
