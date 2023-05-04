// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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
#include "group_quantize_decompress.hpp"

#include <iostream>
namespace popart {

/////////////////////////////////////////////////////////////
////// Fwd op

GroupQuantizeDecompressOp::GroupQuantizeDecompressOp(
    const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> GroupQuantizeDecompressOp::clone() const {
  return std::make_unique<GroupQuantizeDecompressOp>(*this);
}

void GroupQuantizeDecompressOp::setup() {
  auto xInfo = inInfo(0);
  auto groupScaleInfo = inInfo(1);
  auto groupBiasInfo = inInfo(2);

  // check expected shapes
  if (xInfo.rank() != 3) {
    throw error("GroupQuantizeDecompressOp::setup x should have rank 3");
  }

  if (groupScaleInfo.rank() != xInfo.rank()) {
    throw error(
        "GroupQuantizeDecompressOp::setup groupScale should same rank as x");
  }

  if (groupBiasInfo.rank() != xInfo.rank()) {
    throw error(
        "GroupQuantizeDecompressOp::setup groupBias should same rank as x");
  }

  if (groupScaleInfo.shape()[2] != 1) {
    throw error("GroupQuantizeDecompressOp::setup groupScale shape at last "
                "dimension should be 1");
  }

  if (groupBiasInfo.shape()[2] != 1) {
    throw error("GroupQuantizeDecompressOp::setup groupBias shape at last "
                "dimension should be 1");
  }

  if (groupScaleInfo.shape() != groupScaleInfo.shape()) {
    throw error("GroupQuantizeDecompressOp::setup groupScale and groupBias "
                "should have same shape");
  }

  auto n_rows = xInfo.shape()[0];
  auto n_groups = xInfo.shape()[1];
  auto n_ids = xInfo.shape()[2];

  // x decompressed
  outInfo(0) = TensorInfo(groupScaleInfo.data_type(),
                          Shape{n_rows, n_groups * n_ids * 4});
}

void GroupQuantizeDecompressOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
}

} // namespace popart