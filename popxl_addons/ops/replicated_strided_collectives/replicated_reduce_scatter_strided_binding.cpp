// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

/*
For the op to work you need to run the `OpToIdentityPattern` which is called as
part of `PreAliasPatterns` after autodiffing the op.

You also need python header files:
sudo apt install libpython3.6-dev
*/

#include <map>
#include <memory>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/basicoptionals.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>

#include <popart/op/collectives/collectives.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>

#include <poplar/Graph.hpp>
#include <popops/Zero.hpp>
#include <poputil/exceptions.hpp>

#include <gcl/Collectives.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mgcl.hpp"

namespace py = pybind11;

using InMapType  = std::map<popart::InIndex, popart::TensorId>;
using OutMapType = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex   = int;

namespace popart {

#define CUSTOM_OP_DOMAIN "popxl.addons.ops"

// -------------- Op --------------
class ReplicatedReduceScatterStridedOp : public CollectivesBaseOp {
public:
  ReplicatedReduceScatterStridedOp(
      const OperatorIdentifier &_opid,
      const CollectiveOperator &op_,
      const uint32_t stride_,
      const uint32_t groupSize_,
      bool configureOutputForReplicatedTensorSharding_,
      const Op::Settings &settings_)
      : CollectivesBaseOp(_opid, CommGroup(CommGroupType::None, 0), settings_),
        op{op_}, stride(stride_), groupSize(groupSize_),
        configureOutputForReplicatedTensorSharding{
            configureOutputForReplicatedTensorSharding_} {}

  std::unique_ptr<Op> clone() const {
    return std::make_unique<ReplicatedReduceScatterStridedOp>(*this);
  }
  CollectiveOperator getCollectiveOp() const { return op; }

  ReplicatedTensorShardingIndices
  getReplicatedTensorShardingIndices() const override {
    return {{{}, {ReplicatedReduceScatterStridedOp::getOutIndex()}}};
  }
  /**
   * Check \a RTS mode (see collectives.hpp)
   * \return True if this operation is configured for replicated tensor sharding
   */
  bool isConfigureOutputForReplicatedTensorSharding() const override {
    return configureOutputForReplicatedTensorSharding ||
           hasInput(
               ReplicatedReduceScatterStridedOp::getCollectiveLinkedIndex()) ||
           !outInfo(ReplicatedReduceScatterStridedOp::getOutIndex())
                .metaShape()
                .empty();
  }

  void setup() {
    if (!(op == CollectiveOperator::Add || op == CollectiveOperator::Mean)) {
      throw error(
          "Cannot create ReplicatedReduceScatterStridedOp op. "
          "CollectiveOperator::Add and CollectiveOperator::Mean are the "
          "only collective operators "
          "that are currently implemented.");
    }
    if (stride == 0 || groupSize == 0) {
      throw error("Cannot create ReplicatedAllGatherStrided op "
                  "stride and group size must be > 0.");
    }
    const auto &inInfo_ = inInfo(getInIndex());

    auto globalReplicationFactor =
        getIr().getSessionOptions().getGlobalReplicationFactor();
    auto replicationFactor = globalReplicationFactor;
    int64_t nelms          = inInfo_.nelms();
    uint32_t outElms       = std::ceil(float(nelms) / float(groupSize));

    Shape metaShape;
    if (isConfigureOutputForReplicatedTensorSharding()) {
      metaShape = inInfo_.shape();
    }

    outInfo(getOutIndex()) =
        TensorInfo(inInfo_.dataType(), {outElms}, metaShape);

    logging::op::trace("[ReplicatedReduceScatterOp] Global replication factor: "
                       "{}, sharding factor: {}",
                       globalReplicationFactor,
                       replicationFactor);
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &os) const {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("op", op);
    os.appendAttribute("stride", stride);
    os.appendAttribute("groupSize", groupSize);
  }

  static OperatorIdentifier defaultOperatorId() {
    return OperatorIdentifier{
        CUSTOM_OP_DOMAIN,
        "ReplicatedReduceScatterStrided",
        1,      // Op version
        {1, 1}, // number of inputs
        1       // number of outputs
    };
  }

  static ReplicatedReduceScatterStridedOp *
  createOpInGraph(popart::Graph &graph,
                  const InMapType &in,
                  const OutMapType &out,
                  const CollectiveOperator &op,
                  const int32_t stride,
                  const int32_t groupSize,
                  bool configureOutputForReplicatedTensorSharding,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<ReplicatedReduceScatterStridedOp>(
        in,
        out,
        ReplicatedReduceScatterStridedOp::defaultOperatorId(),
        op,
        uint32_t(stride),
        uint32_t(groupSize),
        configureOutputForReplicatedTensorSharding,
        settings);
  }

  uint32_t getStride() const { return stride; }
  int64_t getCommSize() const override { return groupSize; }

protected:
  CollectiveOperator op;
  uint32_t stride;
  uint32_t groupSize;
  /**
   * If enabled, configures the Op for replicated tensor sharding
   */
  bool configureOutputForReplicatedTensorSharding;
};

const popart::OperatorIdentifier ReplicatedReduceScatterStrided =
    ReplicatedReduceScatterStridedOp::defaultOperatorId();

// -------------- OpX --------------
namespace popx {

class ReplicatedReduceScatterStridedOpx : public CollectivesBaseOpx {
public:
  ReplicatedReduceScatterStridedOpx(Op *op, Devicex *devicex)
      : CollectivesBaseOpx(op, devicex) {
    verifyOp<ReplicatedReduceScatterStridedOp>(
        op, {ReplicatedReduceScatterStridedOp::defaultOperatorId()});
  }

  void grow(poplar::program::Sequence &prog) const {
    const auto &rrsOp = getOp<ReplicatedReduceScatterStridedOp>();

    const auto inIndex   = ReplicatedReduceScatterStridedOp::getInIndex();
    auto toReduceScatter = getInTensor(inIndex);

    if (rrsOp.isConfigureOutputForReplicatedTensorSharding()) {
      auto group = getCollectiveLinkedGroup(
          CollectivesBaseOp::getDefaultTensorShardingGroupIndex());

      ViewChangers viewChangers(
          {std::make_shared<ReplicatedGatherInScatterOutViewChanger>(
              outInfo(ReplicatedReduceScatterStridedOp::getOutIndex()).nelms(),
              group.id)});
      setOutViewChangers(ReplicatedReduceScatterStridedOp::getOutIndex(),
                         viewChangers);

      if (!hasInViewChangers(ReplicatedReduceScatterStridedOp::getInIndex()) ||
          getInViewChangers(ReplicatedReduceScatterStridedOp::getInIndex()) !=
              viewChangers) {
        logging::opx::trace(
            "ReplicatedReduceScatterOpx::grow rearranging {}",
            inId(ReplicatedReduceScatterStridedOp::getInIndex()));

        // Tensor not rearranged for reduceScatter yet, do it now
        auto cbr = createCollectiveBalancedReorder(
            toReduceScatter,
            CollectivesBaseOp::getDefaultTensorShardingGroupIndex());
        auto c = cbr->createCollectivesTensor(
            toReduceScatter.elementType(),
            inId(ReplicatedReduceScatterStridedOp::getInIndex()));
        popops::zero(graph(), c, prog, {"zeroScatter"});
        auto ref = cbr->undoRearrangeForCollective(c);
        if (hasInViewChangers(ReplicatedReduceScatterStridedOp::getInIndex())) {
          prog.add(poplar::program::Copy(
              getInViewChangers(ReplicatedReduceScatterStridedOp::getInIndex())
                  .apply(toReduceScatter)
                  .flatten(),
              ref.flatten(),
              false,
              debugContext()));
        } else {
          prog.add(poplar::program::Copy(
              toReduceScatter.flatten(), ref.flatten(), false, debugContext()));
        }
        toReduceScatter = c;
      }
    }
    const poplar::OptionFlags &reduceScatterOptions =
        dv_p->lowering().gclOptions;

    poplar::Tensor reducedScattered = reduceScatterStrided(
        graph(),
        toReduceScatter.flatten(),
        prog,
        getPoplarCollectiveOperator(rrsOp.getCollectiveOp()),
        rrsOp.getStride(),
        rrsOp.getCommSize(),
        debugContext("replicatedReduceScatterStrided"),
        reduceScatterOptions);

    setOutTensor(ReplicatedReduceScatterStridedOp::getOutIndex(),
                 reducedScattered);
  }
};

popx::OpxCreator<ReplicatedReduceScatterStridedOpx>
    ReplicatedReduceScatterStridedOpxCreator(ReplicatedReduceScatterStrided);

} // namespace popx

} // namespace popart

// -------------- PyBind --------------
// `replicated_reduce_scatter_strided_binding` must equal filename
PYBIND11_MODULE(replicated_reduce_scatter_strided_binding, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<popart::ReplicatedReduceScatterStridedOp,
             popart::Op,
             std::shared_ptr<popart::ReplicatedReduceScatterStridedOp>>
      binding(m, "ReplicatedReduceScatterStridedOp");
  binding.def(py::init<const popart::OperatorIdentifier &,
                       const popart::CollectiveOperator &,
                       const uint32_t,
                       const uint32_t,
                       const bool,
                       const popart::Op::Settings &>(),
              py::arg("opid"),
              py::arg("op"),
              py::arg("stride"),
              py::arg("groupSize"),
              py::arg("configureOutputForReplicatedTensorSharding"),
              py::arg("settings"));
  binding.def_static(
      "createOpInGraph",
      py::overload_cast<popart::Graph &,
                        const InMapType &,
                        const OutMapType &,
                        const popart::CollectiveOperator &,
                        const int32_t,
                        const int32_t,
                        bool,
                        const popart::Op::Settings &>(
          &popart::ReplicatedReduceScatterStridedOp::createOpInGraph),
      py::arg("graph"),
      py::arg("inputs"),
      py::arg("outputs"),
      py::arg("op"),
      py::arg("stride"),
      py::arg("groupSize"),
      py::arg("configureOutputForReplicatedTensorSharding"),
      py::arg("settings"),
      py::return_value_policy::reference);
  binding.def("outTensor",
              py::overload_cast<OutIndex>(
                  &popart::ReplicatedReduceScatterStridedOp::outTensor),
              py::return_value_policy::reference);
  binding.def(
      "getGraph",
      py::overload_cast<>(&popart::ReplicatedReduceScatterStridedOp::getGraph),
      py::return_value_policy::reference);
};

// -------------- cppimport --------------
// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare']
cfg['sources'] = ['mgcl.cpp']
cfg['libraries'] = ['popart']
setup_pybind11(cfg)
%>
*/
