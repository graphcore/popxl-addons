// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CROSSENTROPYSHARDED_HPP
#define GUARD_NEURALNET_CROSSENTROPYSHARDED_HPP

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/replicagrouping.hpp>
#include <popart/vendored/optional.hpp>

#include "common.hpp"

namespace popart {

class CrossEntropyShardedOp : public Op {
public:
  CrossEntropyShardedOp(const OperatorIdentifier &_opid, ReplicaGrouping group,
                        float availableMemoryProportion_,
                        const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() final;

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  static CrossEntropyShardedOp *
  createOpInGraph(popart::Graph &graph, const InMapType &in,
                  const OutMapType &out, ReplicaGrouping group,
                  float availableMemoryProportion,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<CrossEntropyShardedOp>(
        in, out, CrossEntropySharded, group, availableMemoryProportion,
        settings);
  }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setAvailableMemoryProportion(
      const nonstd::optional<float> availableMemoryProportion_) {
    availableMemoryProportion = availableMemoryProportion_.value();
  }
  float getAvailableMemoryProportion() { return availableMemoryProportion; }
  ReplicaGrouping getGroup() { return group; }

protected:
  ReplicaGrouping group;
  float availableMemoryProportion;
};

class CrossEntropyShardedGradOp : public Op {
public:
  CrossEntropyShardedGradOp(const CrossEntropyShardedOp &op,
                            float availableMemoryProportion);

  void setup() final;
  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  int getlogSoftmaxStartIndex() const { return logSoftmaxIndex_; }
  int getlogitsStartIndex() const { return logitsIndex_; }
  int getlabelsStartIndex() const { return labelsIndex_; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  void setAvailableMemoryProportion(
      const nonstd::optional<float> availableMemoryProportion_) {
    availableMemoryProportion = availableMemoryProportion_.value();
  }
  float getAvailableMemoryProportion() { return availableMemoryProportion; }

private:
  std::vector<GradInOutMapper> inGradMap;
  std::map<int, int> outGradMap;
  int logSoftmaxIndex_;
  int logitsIndex_;
  int labelsIndex_;

protected:
  float availableMemoryProportion;
};

} // namespace popart

#endif