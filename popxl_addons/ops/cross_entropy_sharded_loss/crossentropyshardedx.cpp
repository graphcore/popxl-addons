// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <snap/Tensor.hpp>
#include <popart/error.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/util.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/exceptions.hpp>

#include <gcl/Collectives.hpp>
#include <poplar/Program.hpp>
#include <poplar/TensorCloneMethod.hpp>
#include <poplar/Type.hpp>
#include <poplin/MatMul.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Fill.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/SelectScalarFromRows.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>

#include "common.hpp"
#include "crossentropysharded.hpp"
#include "crossentropyshardedx.hpp"

#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace popart {
namespace popx {

/////////////////////////////////////////////////////////////
// Utils

/**
 * @brief Get the first tile of a tensor
 *
 * @param g
 * @param t
 * @return std::size_t
 */
std::size_t getFirstTile(poplar::Graph &g, poplar::Tensor t) {
  auto m = g.getTileMapping(t);
  for (auto i = 0u; i < m.size(); ++i) {
    if (!m[i].empty()) {
      return i;
    }
  }

  throw std::runtime_error("Tensor '" + t.getDebugStr() +
                           "' has no tile mapping in this graph.");
}

/////////////////////////////////////////////////////////////
/// Forwards opx

CrossEntropyShardedOpx::CrossEntropyShardedOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<CrossEntropyShardedOp>(op, {CrossEntropySharded});

  availableMemoryProportion =
      getOp<CrossEntropyShardedOp>().getAvailableMemoryProportion();
}

/**
 * Returns the negative stable log softmax from a replica-sharded logits input
 * stableLogSoftmax = x - max(x) - log(sum(exp(x - max(x))))
 *
 * @param graph
 * @param prog
 * @param logits {n_samples, n_classes} sharded on n_classes axis
 * @return poplar::Tensor stable log softmax
 */
poplar::Tensor
CrossEntropyShardedOpx::negLogSoftmax(poplar::Graph &graph,
                                      poplar::program::Sequence &prog,
                                      const poplar::Tensor &logits) const {

  auto elementType = logits.elementType();

  std::vector<poplar::Tensor> maxs;
  auto maxPartial =
      popops::reduce(graph,
                     logits,
                     {1},
                     popops::ReduceParams(popops::Operation::MAX, false),
                     prog,
                     debugContext("max_reduce_logits"));

  // TODO: replace allreduces with inplace varients
  // Obtain max from partial results
  auto max = gcl::allReduceCrossReplica(graph,
                                        maxPartial,
                                        gcl::CollectiveOperator::MAX,
                                        prog,
                                        debugContext("all_reduce_max"));

  auto maxBroadcasted = max.expand({1}).broadcast(logits.dim(1), 1);

  // Sub off the max for stable softmax
  auto translated = popops::sub(
      graph, logits, maxBroadcasted, prog, debugContext("sub_logits_max"));

  // Compute the softmax numerators and partial denominator
  auto numerators =
      popops::exp(graph, translated, prog, debugContext("exp_sub"));
  auto denominatorPartial = popops::reduce(graph,
                                           numerators,
                                           numerators.elementType(),
                                           {1},
                                           popops::Operation::ADD,
                                           prog,
                                           debugContext("reduce_numerators"));

  // All reduce the partial denominators to get global denominator on every IPU:
  auto denominator =
      gcl::allReduceCrossReplica(graph,
                                 denominatorPartial,
                                 gcl::CollectiveOperator::ADD,
                                 prog,
                                 debugContext("all_reduce_denominator"));

  // Final calculation of log softmax on each shard
  auto logDenominator =
      popops::map(graph,
                  popops::expr::Cast(popops::expr::Log(popops::expr::_1),
                                     elementType), // cast(log(denom))
                  {denominator},
                  prog,
                  debugContext("log_denominator"));

  // Make translated -> logSoftmax
  popops::subInPlace(graph,
                     translated,
                     logDenominator.expand({1}),
                     prog,
                     debugContext("stableLogSoftmax"));

  // logSoftmax -> negLogSoftmax
  popops::negInPlace(graph, translated, prog, debugContext("negLogSoftmax"));
  return translated;
}

/**
 *
 *
 * @param graph
 * @param prog
 * @param negLogSoftmax - {n_samples, n_classes} sharded on classes axis
 * @param indices - {n_samples,} identical on each shard. Indices should already
 * be adjusted and so can be negative or out of range (OOR) of sharded classes
 * axis
 * @return poplar::Tensor loss
 */
poplar::Tensor CrossEntropyShardedOpx::takeTrue(poplar::Graph &graph,
                                                poplar::program::Sequence &prog,
                                                poplar::Tensor &negLogSoftmax,
                                                poplar::Tensor &indices) const {
  assert(negLogSoftmax.shape()[0] == indices.shape()[0]);
  auto nSamples = negLogSoftmax.shape()[0];
  auto nClasses = negLogSoftmax.shape()[1]; // sharded number of classes

  auto uIndices = popops::cast(
      graph, indices, poplar::UNSIGNED_INT, prog, debugContext("uIndices"));

  // Obtain loss corresponding to true label (indicated by indices)
  auto sliceOptions = poplar::OptionFlags();
  sliceOptions.set("availableMemoryProportion",
                   std::to_string(availableMemoryProportion));
  sliceOptions.set("usedForSlice", "true");
  sliceOptions.set("usedForUpdate", "false");
  sliceOptions.set("indicesAreSorted", "true");
  auto slicePlan = popops::embedding::plan(graph,
                                           negLogSoftmax.elementType(),
                                           nSamples,
                                           nClasses,
                                           1,
                                           {nClasses - 1}, // max index value
                                           sliceOptions);

  auto lossPartial =
      popops::groupedMultiSlice(graph,
                                negLogSoftmax.expand({2}),
                                uIndices.reshape({nSamples, 1, 1}),
                                {0},
                                {1},
                                prog,
                                slicePlan,
                                sliceOptions,
                                debugContext("lossPartial"))
          .flatten();

  // Zero losses that correspond to out of range (OOR) indices
  namespace pe            = popops::expr;
  auto inRangeIndicesExpr = pe::And(pe::Gte(pe::_2, pe::Const(0)),
                                    pe::Lt(pe::_2, pe::Const(nClasses)));
  auto outputExpr         = pe::TernaryOp(
      pe::TernaryOpType::SELECT, pe::_1, pe::Const(0), inRangeIndicesExpr);
  auto operands = {lossPartial, indices};
  popops::mapInPlace(
      graph, outputExpr, operands, prog, debugContext("zero_OOR_indices"));

  // Allreduce result
  auto loss =
      gcl::allReduceCrossReplica(graph,
                                 lossPartial,
                                 gcl::CollectiveOperator::ADD,
                                 prog,
                                 debugContext("allreduce_partial_losses"));

  return loss;
}

void CrossEntropyShardedOpx::grow(snap::program::Sequence &prog) const {

  auto &graphPop = graph().getPoplarGraph();
  auto &progPop  = prog.getPoplarSequence();

  poplar::Tensor logits  = getInTensor(0).getPoplarTensor();
  poplar::Tensor indices = getInTensor(1).getPoplarTensor();

  auto negLogSoftmax_ = negLogSoftmax(graphPop, progPop, logits);
  auto loss           = takeTrue(graphPop, progPop, negLogSoftmax_, indices);

  // Outputs: loss, negLogSoftmax
  setOutTensor(0, snap::Tensor{loss, graph()});
  setOutTensor(1, snap::Tensor{negLogSoftmax_, graph()});
}

/////////////////////////////////////////////////////////////
///// Grad opx

CrossEntropyShardedGradOpx::CrossEntropyShardedGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<CrossEntropyShardedGradOp>(op, {CrossEntropyShardedGrad});

  availableMemoryProportion =
      getOp<CrossEntropyShardedGradOp>().getAvailableMemoryProportion();

  logSoftmaxIndex =
      getOp<CrossEntropyShardedGradOp>().getlogSoftmaxStartIndex();
  logitsIndex = getOp<CrossEntropyShardedGradOp>().getlogitsStartIndex();
  labelsIndex = getOp<CrossEntropyShardedGradOp>().getlabelsStartIndex();
}

/**
 * Input:
 * - `loss_grad` should be identical across devices
 * - `negLogSoftmax` should be identical across devices
 * - `logits` are sharded across devices (corresponding to different classes)
 * - `labels` should be identical across devices but adjusted for the different
 * vocab ranges
 *
 * Output:
 * - `loss_logits` are sharded across devices (corresponding to different
 * classes)
 */
void CrossEntropyShardedGradOpx::grow(snap::program::Sequence &prog) const {
  auto &graphPop = graph().getPoplarGraph();
  auto &progPop  = prog.getPoplarSequence();

  auto loss_grad     = getInTensor(0).getPoplarTensor();
  auto negLogSoftmax = getInTensor(2).getPoplarTensor();
  auto logits        = getInTensor(3).getPoplarTensor();
  auto labels        = getInTensor(4).getPoplarTensor();

  auto nSamples = negLogSoftmax.shape()[0];
  auto nClasses = negLogSoftmax.shape()[1]; // sharded number of classes

  auto logits_grad = popops::neg(
      graphPop, negLogSoftmax, progPop, debugContext("neg_negLogSoftmax"));
  popops::expInPlace(graphPop, logits_grad, progPop, debugContext("softmax"));
  popops::mulInPlace(graphPop,
                     logits_grad,
                     loss_grad.expand({1}),
                     progPop,
                     debugContext("mul_softmax_scale"));

  auto minusOne = graphPop.addConstant(
      logits_grad.elementType(), {}, -1.f, debugContext("minusOne"));
  graphPop.setTileMapping(minusOne, getFirstTile(graphPop, logits_grad));

  auto uLabels = popops::cast(graphPop,
                              labels,
                              poplar::UNSIGNED_INT,
                              progPop,
                              debugContext("cast_labels"));

  auto sliceOptions = poplar::OptionFlags();
  sliceOptions.set("availableMemoryProportion",
                   std::to_string(availableMemoryProportion));
  sliceOptions.set("usedForSlice", "false");
  sliceOptions.set("usedForUpdate", "true");
  sliceOptions.set("indicesAreSorted", "true");
  auto slicePlan = popops::embedding::plan(graphPop,
                                           logits_grad.elementType(),
                                           nSamples,
                                           nClasses,
                                           1,
                                           {nClasses - 1}, // max index value
                                           sliceOptions);

  popops::groupedMultiUpdateAdd(graphPop,
                                logits_grad.expand({2}),
                                loss_grad.expand({1, 1, 1}),
                                uLabels.expand({1, 1}),
                                minusOne,
                                {0},
                                {1},
                                progPop,
                                slicePlan,
                                sliceOptions,
                                debugContext("groupedMultiUpdateAdd"));

  setOutTensor(0, snap::Tensor{logits_grad, graph()});
}

/////////////////////////////////////////////////////////////

namespace {
popx::OpxCreator<CrossEntropyShardedOpx>
    CrossEntropyShardedOpxCreator(CrossEntropySharded);
popx::OpxCreator<CrossEntropyShardedGradOpx>
    CrossEntropyShardedGradOpxCreator(CrossEntropyShardedGrad);
} // namespace

} // namespace popx
} // namespace popart