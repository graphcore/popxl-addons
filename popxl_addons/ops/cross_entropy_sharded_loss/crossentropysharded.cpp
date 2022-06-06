// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>
#include <popart/graphcoreoperators.hpp>
#include <popart/names.hpp>

#include <memory>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include "common.hpp"
#include "crossentropysharded.hpp"

namespace popart {

/////////////////////////////////////////////////////////////
////// Fwd op

CrossEntropyShardedOp::CrossEntropyShardedOp(const OperatorIdentifier &_opid,
                                             float availableMemoryProportion_,
                                             const Op::Settings &settings_)
    : Op(_opid, settings_),
      availableMemoryProportion(availableMemoryProportion_) {}

std::unique_ptr<Op> CrossEntropyShardedOp::clone() const {
  return std::make_unique<CrossEntropyShardedOp>(*this);
}

std::vector<std::unique_ptr<Op>> CrossEntropyShardedOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<CrossEntropyShardedGradOp>(
      *this, availableMemoryProportion));
  return result;
}

void CrossEntropyShardedOp::setup() {
  auto logitsInfo  = inInfo(0);
  auto indicesInfo = inInfo(1);

  if (logitsInfo.rank() != 2) {
    throw error("CrossEntropyShardedOp::setup logits must be of rank 2. "
                "e.g. (n_samples, sharded_n_classes)");
  }

  if (indicesInfo.rank() != 1) {
    throw error("CrossEntropyShardedOp::setup indices must be of rank 1. "
                "e.g. (n_samples,)");
  }

  if (logitsInfo.shape()[0] != indicesInfo.shape()[0]) {
    throw error("CrossEntropyShardedOp::setup indices and logits must have the "
                "same number of samples. "
                "shapes should be logits=(n_samples, sharded_n_classes); "
                "indices=(n_samples,)");
  }

  auto n_samples = logitsInfo.shape()[0];
  auto n_classes = logitsInfo.shape()[1];

  outInfo(0) = TensorInfo(logitsInfo.data_type(), Shape{n_samples});
  outInfo(1) = TensorInfo(logitsInfo.data_type(), Shape{n_samples, n_classes});
}

void CrossEntropyShardedOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("availableMemoryProportion", availableMemoryProportion);
}

/////////////////////////////////////////////////////////////
////// Grad op

CrossEntropyShardedGradOp::CrossEntropyShardedGradOp(
    const CrossEntropyShardedOp &op,
    float availableMemoryProportion_)
    : Op(CrossEntropyShardedGrad, op.getSettings()),
      availableMemoryProportion(availableMemoryProportion_) {

  // Defines inputs
  logSoftmaxIndex_ = 2;
  logitsIndex_     = 3;
  labelsIndex_     = 4;

  inGradMap.push_back({0, 0, GradOpInType::GradOut}); // dE/dloss
  inGradMap.push_back({1, 0, GradOpInType::Out});     // loss
  inGradMap.push_back(
      {logSoftmaxIndex_, 1, GradOpInType::Out}); // logSoftmax sharded

  inGradMap.push_back({logitsIndex_, 0, GradOpInType::In}); // logits inputs
  inGradMap.push_back({labelsIndex_, 1, GradOpInType::In}); // labels inputs

  // Defines outputs
  outGradMap[0] = 0; // logits grad
}

void CrossEntropyShardedGradOp::setup() { outInfo(0) = inInfo(logitsIndex_); }

std::unique_ptr<Op> CrossEntropyShardedGradOp::clone() const {
  return std::make_unique<CrossEntropyShardedGradOp>(*this);
}

void CrossEntropyShardedGradOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("availableMemoryProportion", availableMemoryProportion);
}

const std::vector<GradInOutMapper> &
CrossEntropyShardedGradOp::gradInputInfo() const {
  return inGradMap;
}

const std::map<int, int> &
CrossEntropyShardedGradOp::gradOutToNonGradIn() const {
  return outGradMap;
}

} // namespace popart