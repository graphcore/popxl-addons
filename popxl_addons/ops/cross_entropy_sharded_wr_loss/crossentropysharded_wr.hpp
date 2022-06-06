// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CROSSENTROPYSHARDEDWR_HPP
#define GUARD_NEURALNET_CROSSENTROPYSHARDEDWR_HPP

#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/vendored/optional.hpp>

#include "common.hpp"

namespace popart {

class CrossEntropyShardedWROp : public Op {
public:
  CrossEntropyShardedWROp(const OperatorIdentifier &_opid,
                          std::vector<int64_t> ipus_,
                          const Op::Settings &settings_);

  std::unique_ptr<Op> clone() const override;
  std::vector<std::unique_ptr<Op>> getGradOps() override;
  void setup() final;

  // Inputs and outputs are variadic
  static InIndex getLogitsStartIndex() { return 0; }
  InIndex getIndicesStartIndex() { return 0 + ipus.size(); }
  static OutIndex getOutStartIndex() { return 0; }

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool canBeReplacedByIdentity() const override;

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex index,
                                   std::set<OpId> &visited) const override;

  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex index,
                                    std::set<OpId> &visited) const override;

  std::vector<int64_t> getIpus() const { return ipus; }

  static CrossEntropyShardedWROp *
  createOpInGraph(popart::Graph &graph,
                  const InMapType &in,
                  const OutMapType &out,
                  const std::vector<int64_t> ipus,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<CrossEntropyShardedWROp>(
        in, out, CrossEntropyShardedWR, ipus, settings);
  }

private:
  const std::vector<int64_t> ipus;
};

class CrossEntropyShardedGradWROp : public Op {
public:
  CrossEntropyShardedGradWROp(const CrossEntropyShardedWROp &op,
                              std::vector<int64_t> ipus_);

  void setup() final;
  std::unique_ptr<Op> clone() const override;
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;

  bool canBeReplacedByIdentity() const override;

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  VGraphIdAndTileSet
  getIntrospectionInVirtualGraphId(InIndex index,
                                   std::set<OpId> &visited) const override;

  VGraphIdAndTileSet
  getIntrospectionOutVirtualGraphId(OutIndex index,
                                    std::set<OpId> &visited) const override;

  std::vector<int64_t> getIpus() const { return ipus; }

  int getlogSoftmaxStartIndex() const { return logSoftmaxIndex_; }
  int getlogitsStartIndex() const { return logitsIndex_; }
  int getlabelsStartIndex() const { return labelsIndex_; }

private:
  const std::vector<int64_t> ipus;
  std::vector<GradInOutMapper> inGradMap;
  std::map<int, int> outGradMap;
  int logSoftmaxIndex_;
  int logitsIndex_;
  int labelsIndex_;
};

} // namespace popart

#endif