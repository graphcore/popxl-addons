// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <cstdint>
#include <layer_norm_distributed.hpp>
#include <map>
#include <memory>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <string>
#include <vector>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/tensorinfo.hpp"
#include <popart/op/collectives/collectives.hpp>

namespace popart {
struct OperatorIdentifier;

LayerNormDistributedOp::LayerNormDistributedOp(const OperatorIdentifier &opid_,
                                               float epsilon_,
                                               const ReplicaGrouping &group,
                                               const Op::Settings &settings_)
    : CollectivesBaseOp(opid_, group, settings_), epsilon(epsilon_) {}

std::unique_ptr<Op> LayerNormDistributedOp::clone() const {
  return std::make_unique<LayerNormDistributedOp>(*this);
}

std::vector<std::unique_ptr<Op>> LayerNormDistributedOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<LayerNormDistributedGradOp>(*this));
  return upops;
}

bool LayerNormDistributedOp::canBeReplacedByIdentity() const {
  return inInfo(getXInIndex()).nelms() == 0;
}

void LayerNormDistributedOp::setup() {
  // The input and output are of shape (N x H).
  outInfo(getYOutIndex()) = inInfo(getXInIndex());

  // For each sample (dimension 0), there is a single mean and a
  // single inverse standard deviation
  outInfo(getInvStdDevOutIndex()) = {inInfo(getXInIndex()).dataType(),
                                     {inInfo(getXInIndex()).dim(0)}};
  outInfo(getMeanOutIndex()) = {inInfo(getXInIndex()).dataType(),
                                {inInfo(getXInIndex()).dim(0)}};
}

void LayerNormDistributedOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("group", getReplicaGrouping());
  os.appendAttribute("epsilon", epsilon);
}

LayerNormDistributedGradOp::LayerNormDistributedGradOp(
    const LayerNormDistributedOp &op_)
    : CollectivesBaseOp(LayerNormDistributedGrad, op_.getReplicaGrouping(),
                        op_.getSettings()),
      epsilon(op_.getEpsilon()),
      fwdInInfo(op_.inInfo(LayerNormDistributedOp::getXInIndex())),
      fwdScaleInInfo(op_.inInfo(LayerNormDistributedOp::getScaleInIndex())),
      fwdBInInfo(op_.inInfo(LayerNormDistributedOp::getBInIndex())) {}

const std::map<int, int> &
LayerNormDistributedGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getXGradOutIndex(), LayerNormDistributedOp::getXInIndex()},
      {getScaleOutIndex(), LayerNormDistributedOp::getScaleInIndex()},
      {getBOutIndex(), LayerNormDistributedOp::getBInIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &
LayerNormDistributedGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getYGradInIndex(), LayerNormDistributedOp::getYOutIndex(),
       GradOpInType::GradOut},
      {getXInIndex(), LayerNormDistributedOp::getXInIndex(), GradOpInType::In},
      {getScaleInIndex(), LayerNormDistributedOp::getScaleInIndex(),
       GradOpInType::In},
      {getMeanInIndex(), LayerNormDistributedOp::getMeanOutIndex(),
       GradOpInType::Out},
      {getInvStdDevInIndex(), LayerNormDistributedOp::getInvStdDevOutIndex(),
       GradOpInType::Out}};
  return inInfo;
}

void LayerNormDistributedGradOp::setup() {
  TensorInfo xOutInfo = fwdInInfo;
  xOutInfo.set(xOutInfo.dataType(), inTensor(getXInIndex())->info.shape());

  outInfo(getXGradOutIndex()) = xOutInfo;
  outInfo(getScaleOutIndex()) = fwdScaleInInfo;
  outInfo(getBOutIndex()) = fwdBInInfo;
}

std::unique_ptr<Op> LayerNormDistributedGradOp::clone() const {
  return std::make_unique<LayerNormDistributedGradOp>(*this);
}

void LayerNormDistributedGradOp::appendOutlineAttributes(
    OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("epsilon", epsilon);
  os.appendAttribute("group", getReplicaGrouping());
}
namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition layerNormDistributedOpDef(
    {OpDefinition::Inputs({
         {"X", T},
         {"Scale", T},
         {"Bias", T},
     }),
     OpDefinition::Outputs({{"Y", T}, {"Mean", T}, {"Var", T}}),
     OpDefinition::Attributes({{"epsilon", {"*"}}, {"group", {"*"}}})});

static OpCreator<LayerNormDistributedOp> layerNormDistributedOpCreator(
    OpDefinitions({
        {LayerNormDistributed, layerNormDistributedOpDef},
    }),
    [](const OpCreatorInfo &info) {
      // default epsilon is 10**(-5)
      float epsilon =
          info.attributes.getAttribute<Attributes::Float>("epsilon", 1e-5f);

      const ReplicaGrouping &group = extractReplicaGroupingFromVector(
          info.attributes.getAttribute<Attributes::Ints>("group"));

      return std::unique_ptr<Op>(
          new LayerNormDistributedOp(info.opid, epsilon, group, info.settings));
    },
    true);

} // namespace

} // namespace popart