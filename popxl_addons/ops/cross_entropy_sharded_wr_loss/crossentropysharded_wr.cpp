// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <popart/graphcoreoperators.hpp>
#include <popart/names.hpp>
#include <string>
#include <vector>

#include <memory>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include "common.hpp"
#include "crossentropysharded_wr.hpp"

namespace popart {

/////////////////////////////////////////////////////////////
////// Fwd op

CrossEntropyShardedWROp::CrossEntropyShardedWROp(
    const OperatorIdentifier &_opid, const std::vector<int64_t> ipus_,
    const Op::Settings &settings_)
    : Op(_opid, settings_), ipus(ipus_) {}

std::unique_ptr<Op> CrossEntropyShardedWROp::clone() const {
  return std::make_unique<CrossEntropyShardedWROp>(*this);
}

std::vector<std::unique_ptr<Op>> CrossEntropyShardedWROp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<CrossEntropyShardedGradWROp>(*this, ipus));
  return result;
}

void CrossEntropyShardedWROp::setup() {
  int64_t numShards = ipus.size();
  auto numInputs = input->n();
  auto numOutputs = output->n();

  if (numInputs == 0) {
    throw error(
        "CrossEntropyShardedWROp::setup there must be at least 2 inputs.");
  }

  if (numOutputs != 1 + numShards) {
    throw error(
        "CrossEntropyShardedWROp::setup number of outputs should be 1 + "
        "numShards.");
  }

  if (numInputs != 2 * ipus.size()) {
    throw error(
        "CrossEntropyShardedWROp::setup number of inputs does not equal "
        "twice the number of "
        "ipus.");
  }

  auto logitsInfo0 = inInfo(0);
  if (logitsInfo0.rank() != 2) {
    throw error("CrossEntropyShardedWROp::setup all logits must be of rank 3. "
                "e.g. (n_samples, sub_n_classes)");
  }

  auto nSamples = logitsInfo0.shape()[0];

  for (int i = 0; i < numShards; i++) {
    auto t = inInfo(i);
    if (t.data_type() != logitsInfo0.data_type()) {
      throw error(
          "CrossEntropyShardedWROp::setup not all logits inputs have the "
          "same datatype.");
    }
    if (t.rank() != 2) {
      throw error(
          "CrossEntropyShardedWROp::setup all logits must be of rank 2. "
          "e.g. (n_samples, sub_n_classes)");
    }
    if (t.shape()[0] != nSamples) {
      throw error(
          "CrossEntropyShardedWROp::setup all logits must have the same "
          "number of samples.");
    }
  }

  auto indiciesInfo0 = inInfo(numShards);

  for (int i = numShards; i < numInputs; i++) {
    auto t = inInfo(i);
    if (t.data_type() != indiciesInfo0.data_type()) {
      throw error("CrossEntropyShardedWROp::setup not all indices inputs have "
                  "the same datatype.");
    }
    if (t.rank() != 1) {
      throw error(
          "CrossEntropyShardedWROp::setup all indices must be of rank 1. "
          "e.g. (n_samples,)");
    }
    if (t.shape()[0] != nSamples) {
      throw error("CrossEntropyShardedWROp::setup all indices and logits must "
                  "have the same token number");
    }
  }

  outInfo(0) = TensorInfo(logitsInfo0.data_type(), Shape{nSamples});
  for (int i = 0; i < numShards; i++) {
    auto nClasses = inInfo(i).shape()[1];
    outInfo(i + 1) =
        TensorInfo(logitsInfo0.data_type(), Shape{nSamples, nClasses});
  }
}

void CrossEntropyShardedWROp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("ipus", ipus);
}

bool CrossEntropyShardedWROp::canBeReplacedByIdentity() const { return false; }

VGraphIdAndTileSet CrossEntropyShardedWROp::getIntrospectionInVirtualGraphId(
    InIndex index, std::set<OpId> &visited) const {
  index %= ipus.size();
  return {ipus.at(index), settings.tileSet};
}

VGraphIdAndTileSet CrossEntropyShardedWROp::getIntrospectionOutVirtualGraphId(
    OutIndex index, std::set<OpId> &visited) const {
  if (index == 0) {
    return {ipus.at(0), settings.tileSet};
  }
  index -= 1;
  return {ipus.at(index), settings.tileSet};
}

/////////////////////////////////////////////////////////////
////// Grad op

CrossEntropyShardedGradWROp::CrossEntropyShardedGradWROp(
    const CrossEntropyShardedWROp &op, std::vector<int64_t> ipus_)
    : Op(CrossEntropyShardedGradWR, op.getSettings()), ipus(ipus_) {

  auto numShards = ipus_.size();

  // Defines inputs
  logSoftmaxIndex_ = 2;
  logitsIndex_ = logSoftmaxIndex_ + numShards;
  labelsIndex_ = logitsIndex_ + numShards;

  inGradMap.push_back({0, 0, GradOpInType::GradOut}); // dE/dloss
  inGradMap.push_back({1, 0, GradOpInType::Out});     // loss

  for (int i = 0; i < numShards; i++) {
    inGradMap.push_back(
        {logSoftmaxIndex_ + i, 1 + i, GradOpInType::Out}); // logSoftmax sharded
  }

  for (int i = 0; i < 2 * numShards; i++) {
    inGradMap.push_back(
        {logitsIndex_ + i, i, GradOpInType::In}); // logits and labels inputs
  }

  // Defines outputs
  for (int i = 0; i < numShards; i++) {
    outGradMap[i] = i; // logits grad
  }
}

void CrossEntropyShardedGradWROp::setup() {
  for (int i = 0; i < ipus.size(); i++) {
    outInfo(i) = inInfo(logitsIndex_ + i);
  }
}

std::unique_ptr<Op> CrossEntropyShardedGradWROp::clone() const {
  return std::make_unique<CrossEntropyShardedGradWROp>(*this);
}

const std::vector<GradInOutMapper> &
CrossEntropyShardedGradWROp::gradInputInfo() const {
  return inGradMap;
}

const std::map<int, int> &
CrossEntropyShardedGradWROp::gradOutToNonGradIn() const {
  return outGradMap;
}

void CrossEntropyShardedGradWROp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("ipus", ipus);
}

bool CrossEntropyShardedGradWROp::canBeReplacedByIdentity() const {
  return false;
}

VGraphIdAndTileSet
CrossEntropyShardedGradWROp::getIntrospectionInVirtualGraphId(
    InIndex index, std::set<OpId> &visited) const {
  if (index < logSoftmaxIndex_) {
    return {ipus.at(0), settings.tileSet};
  }
  auto index_ = index - logSoftmaxIndex_;
  index_ %= ipus.size();
  return {ipus.at(index_), settings.tileSet};
}

VGraphIdAndTileSet
CrossEntropyShardedGradWROp::getIntrospectionOutVirtualGraphId(
    OutIndex index, std::set<OpId> &visited) const {
  index %= ipus.size();
  return {ipus.at(index), settings.tileSet};
}

} // namespace popart