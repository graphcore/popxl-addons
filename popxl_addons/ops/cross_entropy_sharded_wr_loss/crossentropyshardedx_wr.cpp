// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <iostream>
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
#include <poplar/TensorCloneMethod.hpp>
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
#include "crossentropysharded_wr.hpp"
#include "crossentropyshardedx_wr.hpp"

#include <sstream>
#include <string>
#include <vector>

namespace popart {
namespace popx {

/////////////////////////////////////////////////////////////
// Utils

std::vector<poplar::Tensor> toShards(poplar::Tensor t, int numShards) {
  std::vector<poplar::Tensor> output;
  // Split result (each split is a copy on a different IPU)
  for (int i = 0; i < numShards; i++) {
    output.push_back(t.slice(i, i + 1, 0).squeeze({0}));
  }
  return output;
}

poplar::Tensor
zerosLike(poplar::Graph &g, poplar::Tensor t, poplar::program::Sequence &p) {
  auto z = g.clone(t);
  popops::fill(g, z, p, 0.f);
  return z;
}

std::vector<std::size_t> getPartitionSizes(const std::vector<poplar::Tensor> &t,
                                           std::size_t partitionedAxis) {
  std::vector<std::size_t> partitionSizes;
  for (const auto &p : t) {
    partitionSizes.push_back(p.dim(partitionedAxis));
  }
  return partitionSizes;
}

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

std::pair<poplar::Tensor, std::vector<poplar::Tensor>>
adjustIndicesAndCreateMasks(
    poplar::Graph &graph,
    std::vector<std::reference_wrapper<poplar::Graph>> vGraphs,
    std::vector<std::size_t> &partitionSizes,
    poplar::Tensor indices,
    poplar::program::Sequence &prog,
    const std::string &debug_prefix) {
  auto unsignedIndices = poputil::duplicate(graph, indices, prog);
  std::vector<poplar::Tensor> masks;
  std::size_t partitionCum = 0;
  for (auto s = 0; s < vGraphs.size(); s++) {
    auto &vGraph        = vGraphs[s].get();
    auto shardSuffixStr = std::to_string(s);
    // Create shifted indices on each partition where we have subtracted
    // the partition offset from the labels.
    auto size = partitionSizes[s];
    auto zero = vGraph.addConstant(unsignedIndices.elementType(), {}, 0u);
    auto partitionSize =
        vGraph.addConstant(unsignedIndices.elementType(), {}, size);
    auto partitionStart =
        vGraph.addConstant(unsignedIndices.elementType(), {}, partitionCum);
    auto firstTileOfInput = getFirstTile(vGraph, unsignedIndices[s]);
    vGraph.setTileMapping(partitionStart, firstTileOfInput);
    vGraph.setTileMapping(zero, firstTileOfInput);
    vGraph.setTileMapping(partitionSize, firstTileOfInput);
    popops::subInPlace(vGraph,
                       unsignedIndices[s],
                       partitionStart,
                       prog,
                       debug_prefix + "/shift_" + shardSuffixStr);

    partitionCum += size;

    // From the shifted labels create a mask on each shard to remove labels < 0
    // or >= labels-per-shard (we need this because IPU gathers do not return 0
    // for invalid indices).
    namespace pe     = popops::expr;
    auto betweenExpr = pe::And(pe::Gte(pe::_1, pe::_2), pe::Lt(pe::_1, pe::_3));
    auto operands    = {unsignedIndices[s], zero, partitionSize};
    auto mask        = popops::map(vGraph,
                            betweenExpr,
                            operands,
                            prog,
                            debug_prefix + "/generate_mask_" + shardSuffixStr);
    masks.push_back(mask);
  }

  return std::make_pair(unsignedIndices, masks);
}

poplar::Tensor
copyToAll(poplar::program::Sequence &prog,
          poplar::Graph &graph,
          std::vector<std::reference_wrapper<poplar::Graph>> vGraphs,
          poplar::Tensor input,
          int64_t inputIpu,
          const std::string &debug_prefix) {

  // Copy to all shards, expand first dim so it acts as the shard/IPU index:
  std::vector<poplar::Tensor> copies;
  auto inputTileMapping = vGraphs[inputIpu].get().getTileMapping(input);
  for (auto i = 0; i < vGraphs.size(); ++i) {
    if (i == inputIpu) {
      copies.push_back(input.expand({0}));
    } else {
      auto t = vGraphs[inputIpu].get().clone(input);
      vGraphs[i].get().setTileMapping(t, inputTileMapping);
      copies.push_back(t.expand({0}));
      prog.add(poplar::program::Copy(input, t));
    }
  }

  auto output = poplar::concat(copies, 0);
  return output;
}

std::vector<std::reference_wrapper<poplar::Graph>>
CrossEntropyShardedWROpx::getVGraphs() const {
  auto &op = getOp<CrossEntropyShardedWROp>();

  std::vector<std::reference_wrapper<poplar::Graph>> vGraphs;
  for (auto i = 0; i < numShards; i++) {
    auto emptySet         = std::set<OpId>();
    auto vGraphAndTileSet = op.getIntrospectionInVirtualGraphId(i, emptySet);
    auto &vGraph =
        dv_p->lowering()
            .getVirtualGraph(vGraphAndTileSet.first, vGraphAndTileSet.second)
            .getPoplarGraph();
    vGraphs.push_back(vGraph);
  }
  return vGraphs;
}

std::vector<std::reference_wrapper<poplar::Graph>>
CrossEntropyShardedGradWROpx::getVGraphs() const {
  auto &op = getOp<CrossEntropyShardedGradWROp>();

  std::vector<std::reference_wrapper<poplar::Graph>> vGraphs;
  for (auto i = logitsStartIndex; i < numShards + logitsStartIndex; i++) {
    auto emptySet         = std::set<OpId>();
    auto vGraphAndTileSet = op.getIntrospectionInVirtualGraphId(i, emptySet);
    auto &vGraph =
        dv_p->lowering()
            .getVirtualGraph(vGraphAndTileSet.first, vGraphAndTileSet.second)
            .getPoplarGraph();
    vGraphs.push_back(vGraph);
  }
  return vGraphs;
}

/////////////////////////////////////////////////////////////
/// Forwards opx

CrossEntropyShardedWROpx::CrossEntropyShardedWROpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<CrossEntropyShardedWROp>(op, {CrossEntropyShardedWR});

  numShards = getOp<CrossEntropyShardedWROp>().getIpus().size();
}

// stableLogSoftmax = x - max(x) - log(sum(exp(x - max(x))))
std::vector<poplar::Tensor> CrossEntropyShardedWROpx::sharded_log_softmax(
    poplar::Graph &graph,
    std::vector<std::reference_wrapper<poplar::Graph>> vGraphs,
    poplar::program::Sequence &prog,
    const std::vector<poplar::Tensor> &logits) const {

  auto elementType = logits[0].elementType();
  auto rank        = logits[0].rank();

  std::vector<std::size_t> totalShape(rank, 0);
  for (auto j = 0; j < rank; j++) {
    std::size_t tSize = 0;
    for (auto i = 0; i < numShards; i++) {
      tSize += logits[i].shape()[j];
    }
    totalShape[j] = tSize;
  }

  std::vector<poplar::Tensor> maxs;
  for (auto i = 0; i < numShards; i++) {
    auto max =
        popops::reduce(vGraphs[i],
                       logits[i],
                       {1},
                       popops::ReduceParams(popops::Operation::MAX, false),
                       prog,
                       "/reduce_partial_max");
    // Expand partial results so we can concat them ready for all reduce:
    maxs.push_back(max.expand({0}));
  }

  // All reduce the partial maximums:
  auto interIpuMaxs = poplar::concat(maxs, 0);
  auto max          = gcl::allReduceWithinReplica(graph,
                                         interIpuMaxs,
                                         gcl::CollectiveOperator::MAX,
                                         prog,
                                         "all_reduce_max");

  // Now we can compute the softmax numerators and partial denominator on each
  // shard:
  auto maxSharded = toShards(max, numShards);
  std::vector<poplar::Tensor> shardedDenominators;
  std::vector<poplar::Tensor> shardedTranslated;
  for (auto i = 0; i < numShards; i++) {
    auto logit = logits[i];
    auto max   = maxSharded[i].expand({1}).broadcast(logit.dim(1), 1);

    // Sub off the max for stable softmax:
    auto translated =
        popops::sub(vGraphs[i], logit, max, prog, "partial_sub_max");
    auto numerators = popops::exp(vGraphs[i], translated, prog, "partial_exp");
    auto partialDenominator = popops::reduce(vGraphs[i],
                                             numerators,
                                             numerators.elementType(),
                                             {1},
                                             popops::Operation::ADD,
                                             prog,
                                             "reduce_partial_denom");
    prog.add(poplar::program::WriteUndef(numerators));
    shardedTranslated.push_back(translated);
    shardedDenominators.push_back(partialDenominator.expand({0}));
  }

  // All reduce the partial denominators to get global denominator on every IPU:
  auto interIpuDenom = poplar::concat(shardedDenominators, 0);
  auto denom         = gcl::allReduceWithinReplica(graph,
                                           interIpuDenom,
                                           gcl::CollectiveOperator::ADD,
                                           prog,
                                           "all_reduce_denominator");

  // Final calculation of log softmax on each shard
  for (int i = 0; i < shardedTranslated.size(); i++) {
    auto denomShard = denom[i];
    auto logsum =
        popops::map(vGraphs[i],
                    popops::expr::Cast(popops::expr::Log(popops::expr::_1),
                                       elementType), // cast(log(denom))
                    {denomShard},
                    prog,
                    "log_denominator");
    popops::subInPlace(vGraphs[i],
                       shardedTranslated[i],
                       logsum.expand({1}),
                       prog,
                       "subtract_logsum");
  }
  return shardedTranslated;
}

poplar::Tensor CrossEntropyShardedWROpx::sharded_take_last(
    poplar::Graph &graph,
    std::vector<std::reference_wrapper<poplar::Graph>> vGraphs,
    poplar::program::Sequence &prog,
    std::vector<poplar::Tensor> negLogSoftmax,
    poplar::Tensor indiciesConcat) const {
  // negLogSoftmax = {n_samples, n_classes} // sharded by ipu
  // indiciesConcat = {nIPUs, n_samples} // between 0 and n_classes
  // indices to be shareded copied

  // Input is sharded in the axis that we need to gather from so we must
  // explcitly restrict gathers to shards so that we do not exchange the vocab
  // dimension between IPUs:

  auto partitionSizes = getPartitionSizes(negLogSoftmax, 1);
  poplar::Tensor indices;
  std::vector<poplar::Tensor> masks;
  auto indiciesConcat_     = indiciesConcat.expand({2});
  std::tie(indices, masks) = adjustIndicesAndCreateMasks(
      graph, vGraphs, partitionSizes, indiciesConcat_, prog, "/adjust_indices");

  std::vector<poplar::Tensor> results;
  for (auto s = 0; s < numShards; s++) {
    auto shardSuffixStr = std::to_string(s);
    auto shardInput     = negLogSoftmax[s];

    // TODO: We manually serialise the slicing over rows as
    // multiSlice goes horribly OOM otherwise but this is very
    // slow:
    std::vector<poplar::Tensor> scalars;
    for (auto r = 0u; r < shardInput.dim(0); ++r) {
      auto rowSuffix = std::to_string(r);
      auto row       = shardInput[r];
      auto rowIndex  = indices[s][r].expand({1});

      // Copy the row to a sliceable tensor:
      auto sliceableRow = popops::createSliceableTensor(
          vGraphs[s], row.elementType(), row.shape(), {0}, {1}, {}, {});
      prog.add(poplar::program::Copy(row, sliceableRow, false));

      auto scalar = popops::multiSlice(vGraphs[s],
                                       sliceableRow,
                                       rowIndex,
                                       {0},
                                       {1},
                                       prog,
                                       {},
                                       {},
                                       "take_along_row_" + rowSuffix +
                                           "_shard_" + shardSuffixStr);
      scalars.push_back(scalar);
    }

    auto shardSlice = poplar::concat(scalars);

    // Need to mask the result at invalid indices:
    auto mask  = masks[s];
    auto zeros = zerosLike(vGraphs[s], shardSlice, prog);
    popops::selectInPlace(vGraphs[s],
                          shardSlice,
                          zeros,
                          mask,
                          prog,
                          "/apply_shard_mask_" + shardSuffixStr);

    results.push_back(shardSlice.expand({0}));
  }

  // Reduce the partial masked gathered elements across shards to get the final
  // result onto single IPU for final loss reduction stage.
  auto partials    = poplar::concat(results, 0);
  auto result      = popops::reduce(graph,
                               partials,
                               partials.elementType(),
                               {0},
                               popops::Operation::ADD,
                               prog,
                               "reduce_partial_denom");
  auto result_flat = result.flatten();

  return result_flat;
}

void CrossEntropyShardedWROpx::grow(snap::program::Sequence &prog) const {

  auto &graphTop = topLevelGraph().getPoplarGraph();
  auto vGraphs   = getVGraphs();
  auto &progPop  = prog.getPoplarSequence();

  std::vector<poplar::Tensor> logits;
  std::vector<poplar::Tensor> indices;
  for (auto i = 0; i < numShards; i++) {
    logits.push_back(getInTensor(i).getPoplarTensor());
    indices.push_back(getInTensor(i + numShards).getPoplarTensor().expand({0}));
  }

  auto indiciesConcat = poplar::concat(indices, 0);

  auto logSoftmaxSharded =
      sharded_log_softmax(graphTop, vGraphs, progPop, logits);
  for (int i = 0; i < logSoftmaxSharded.size(); i++) {
    popops::negInPlace(graphTop,
                       logSoftmaxSharded[i],
                       progPop,
                       "/neg_log_softmax" + std::to_string(i));
  }

  auto loss = sharded_take_last(
      graphTop, vGraphs, progPop, logSoftmaxSharded, indiciesConcat);

  setOutTensor(0, snap::Tensor{loss, dstVirtualGraph(0)});
  for (int i = 0; i < numShards; i++) {
    setOutTensor(1 + i, snap::Tensor{logSoftmaxSharded[i], dstVirtualGraph(i)});
  }
}

/////////////////////////////////////////////////////////////
///// Grad opx

CrossEntropyShardedGradWROpx::CrossEntropyShardedGradWROpx(Op *op,
                                                           Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<CrossEntropyShardedGradWROp>(op, {CrossEntropyShardedGradWR});

  ipus      = getOp<CrossEntropyShardedGradWROp>().getIpus();
  numShards = getOp<CrossEntropyShardedGradWROp>().getIpus().size();
  logSoftmaxStartIndex =
      getOp<CrossEntropyShardedGradWROp>().getlogSoftmaxStartIndex();
  logitsStartIndex = getOp<CrossEntropyShardedGradWROp>().getlogitsStartIndex();
  labelsStartIndex = getOp<CrossEntropyShardedGradWROp>().getlabelsStartIndex();
}

// Fused backwards pass is much simpler and more efficient:
void CrossEntropyShardedGradWROpx::grow(snap::program::Sequence &prog) const {
  // gradient on IPU0
  // log_softmax_output is sharded across IPUs
  auto &graphTop = topLevelGraph().getPoplarGraph();
  auto vGraphs   = getVGraphs();
  auto &progPop  = prog.getPoplarSequence();

  // The fwd_inputs for sharded_take_last were -log(softmax()) forwards outputs
  // which we stashed additionally:
  auto gradient = getInTensor(0).getPoplarTensor(); // gradient input (on IPU0)

  std::vector<poplar::Tensor> indicesSharded;
  for (int i = 0; i < numShards; i++) {
    indicesSharded.push_back(
        getInTensor(labelsStartIndex + i).getPoplarTensor().expand({0}));
  }
  auto indices = poplar::concat(indicesSharded, 0).expand({2});

  // We need to scatter-subtract the incoming gradient from what we have
  // computed: The incoming gradient is not sharded as it was reduced for the
  // final loss. The stashed activations are sharded across all IPUs in the
  // dimension we need to update (which could be very large). So like the FWD
  // pass we want to split the scatter operation across shards to avoid any
  // communication of the large dimension.
  // 1. Broadcast the incoming gradient to all shards. This grad is relatively
  // small as it is result of
  //    sparse op (it is going to be scattered into the largest axis of the
  //    outgoing grad tensor).
  auto gradientShardedCopies = copyToAll(progPop,
                                         graphTop,
                                         vGraphs,
                                         gradient,
                                         ipus[0],
                                         "/broadcast_grad_to_shards");

  std::vector<poplar::Tensor> activationsSharded;
  for (auto i = 0u; i < numShards; ++i) {
    // In the forward op we negated the sharded_log_softmax_grad result
    // but we saved the +ve activations so negate the activations:
    auto logSoftmax = getInTensor(logSoftmaxStartIndex + i).getPoplarTensor();
    auto softmax    = popops::neg(vGraphs[i], logSoftmax, progPop);
    popops::expInPlace(vGraphs[i], softmax, progPop, "/recover_softmax_acts");

    auto scale = gradientShardedCopies[i].expand({1});
    popops::mulInPlace(
        vGraphs[i], softmax, scale, progPop, "/scale_softmax_acts");

    activationsSharded.push_back(softmax);
  }

  // 2. Each shard now contains all the grads and all the indices, and a part of
  // the activations 's'. We only want to update indices in s that reside on
  // each shard, so first we need to make adjustments to the indices.
  auto partitionSizes = getPartitionSizes(activationsSharded, 1);
  poplar::Tensor adjustedIndices;
  std::vector<poplar::Tensor> masks;
  std::tie(adjustedIndices, masks) = adjustIndicesAndCreateMasks(
      graphTop, vGraphs, partitionSizes, indices, progPop, "/adjust_indices");

  // 3. Use adjusted indices to update each shard's partition of s in parallel:
  for (auto c = 0u; c < vGraphs.size(); ++c) {
    auto shardSuffixStr = std::to_string(c);
    auto activation     = activationsSharded[c];
    auto grad           = gradientShardedCopies[c];
    auto minusOne = graphTop.addConstant(activation.elementType(), {}, -1.f);
    graphTop.setTileMapping(minusOne, getFirstTile(graphTop, activation));

    // TODO: this is slow to avoid OOM (same as fwds pass):
    for (auto i = 0u; i < activation.dim(0); ++i) {
      auto row        = activation[i];
      auto updateable = popops::createSliceableTensor(
          vGraphs[c], row.elementType(), row.shape(), {0}, {1}, {}, {});
      progPop.add(poplar::program::Copy(row, updateable));

      auto u_ = updateable.expand({1});
      auto g_ = grad[i].expand({0, 0, 0});
      auto a_ = adjustedIndices[c][i].expand({0});

      popops::multiUpdateAdd(vGraphs[c],
                             u_,
                             g_,
                             a_,
                             minusOne,
                             {0},
                             {1},
                             progPop,
                             {},
                             {},
                             "/scatter_gradients_" + shardSuffixStr + "_" +
                                 std::to_string(i));

      // Copy result back:
      progPop.add(poplar::program::Copy(updateable, row, false));
    }
  }

  for (int i = 0; i < numShards; i++) {
    auto t = activationsSharded[i];
    setOutTensor(i, snap::Tensor{t, dstVirtualGraph(i)});
  }
}

/////////////////////////////////////////////////////////////

namespace {
popx::OpxCreator<CrossEntropyShardedWROpx>
    CrossEntropyShardedWROpxCreator(CrossEntropyShardedWR);
popx::OpxCreator<CrossEntropyShardedGradWROpx>
    CrossEntropyShardedGradWROpxCreator(CrossEntropyShardedGradWR);
} // namespace

} // namespace popx
} // namespace popart
