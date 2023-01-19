// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OP_LAYERNORMDISTRIBUTED_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OP_LAYERNORMDISTRIBUTED_HPP_

#include <cstdint>
#include <map>
#include <memory>
#include <popart/op.hpp>
#include <vector>

#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/vendored/optional.hpp>

#include "common.hpp"

namespace popart {
class OpSerialiserBase;
struct OperatorIdentifier;

// TODO: add replica grouping
class LayerNormDistributedOp : public CollectivesBaseOp {
public:
  LayerNormDistributedOp(const OperatorIdentifier &opid_, float epsilon_,
                         const ReplicaGrouping &group,
                         const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static LayerNormDistributedOp *
  createOpInGraph(popart::Graph &graph, const InMapType &in,
                  const OutMapType &out, float epsilon_,
                  const ReplicaGrouping &group,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<LayerNormDistributedOp>(
        in, out, LayerNormDistributed, epsilon_, group, settings);
  }

  // Input's
  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getBInIndex() { return 2; }

  // Ouput's
  static OutIndex getYOutIndex() { return 0; }
  static OutIndex getMeanOutIndex() { return 1; }
  static OutIndex getInvStdDevOutIndex() { return 2; }

  // Attributes
  float getEpsilon() const { return epsilon; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool isNorm() const override { return true; }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool canShard() const override { return true; }

  bool canBeReplacedByIdentity() const final;

private:
  float epsilon;
  // const ReplicaGrouping &group;
};

class LayerNormDistributedGradOp : public CollectivesBaseOp {
public:
  LayerNormDistributedGradOp(const LayerNormDistributedOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;
  std::unique_ptr<Op> clone() const final;

  static InIndex getXInIndex() { return 0; }
  static InIndex getScaleInIndex() { return 1; }
  static InIndex getMeanInIndex() { return 2; }
  static InIndex getInvStdDevInIndex() { return 3; }
  static InIndex getYGradInIndex() { return 4; }

  float getEpsilon() const { return epsilon; }

  static OutIndex getXGradOutIndex() { return 0; }
  static OutIndex getScaleOutIndex() { return 1; }
  static OutIndex getBOutIndex() { return 2; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  bool canShard() const override { return true; }

private:
  float epsilon;
  // const ReplicaGrouping &group;
  TensorInfo fwdInInfo, fwdScaleInInfo, fwdBInInfo;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OP_GROUPNORM_HPP_
