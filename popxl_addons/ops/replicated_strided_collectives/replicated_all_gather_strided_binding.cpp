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
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/popx/op/collectives/collectivesx.hpp>
#include <popart/popx/op/collectives/replicatedallgatherx.hpp>

#include <poplar/Graph.hpp>
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
class ReplicatedAllGatherStridedOp : public CollectivesBaseOp {
public:
  ReplicatedAllGatherStridedOp(const OperatorIdentifier &_opid,
                               const uint32_t stride_,
                               const uint32_t groupSize_,
                               const Op::Settings &settings_)
      : CollectivesBaseOp(_opid, CommGroup(CommGroupType::None, 0), settings_),
        stride(stride_), groupSize(groupSize_) {}

  std::unique_ptr<Op> clone() const {
    return std::make_unique<ReplicatedAllGatherStridedOp>(*this);
  }

  void setup() {
    if (stride == 0 || groupSize == 0) {
      throw error("Cannot create ReplicatedAllGatherStrided op "
                  "stride and group size must be > 0.");
    }
    auto globalReplicationFactor =
        getIr().getSessionOptions().getGlobalReplicationFactor();
    auto replicationFactor = globalReplicationFactor;
    DataType type =
        inTensor(ReplicatedAllGatherStridedOp::getInIndex())->info.dataType();
    Shape shape = gatheredOutInfo.shape();
    if (gatheredOutInfo.shape().empty()) {
      gatheredOutInfo = inInfo(ReplicatedAllGatherStridedOp::getInIndex());
      Shape new_shape(1, groupSize * gatheredOutInfo.nelms());
      shape = new_shape;
    }
    gatheredOutInfo.set(type, shape);
    outInfo(getOutIndex()) = gatheredOutInfo;

    logging::op::trace(
        "[ReplicatedAllGatherStridedOp] Global replication factor: {}, "
        "sharding factor: {}, stride {}, group size {}",
        globalReplicationFactor,
        replicationFactor,
        stride,
        groupSize);
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &os) const {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("stride", stride);
    os.appendAttribute("groupSize", groupSize);
  }

  ReplicatedTensorShardingIndices getReplicatedTensorShardingIndices() const {
    return {{{ReplicatedAllGatherOp::getInIndex()}, {}}};
  }

  bool isConfigureOutputForReplicatedTensorSharding() const {
    return hasInput(ReplicatedAllGatherOp::getCollectiveLinkedIndex()) ||
           !inInfo(ReplicatedAllGatherOp::getInIndex()).metaShape().empty();
  }

  static OperatorIdentifier defaultOperatorId() {
    return OperatorIdentifier{
        CUSTOM_OP_DOMAIN,
        "ReplicatedAllGatherStrided",
        1,      // Op version
        {1, 1}, // number of inputs
        1       // number of outputs
    };
  }

  static ReplicatedAllGatherStridedOp *
  createOpInGraph(popart::Graph &graph,
                  const InMapType &in,
                  const OutMapType &out,
                  const int32_t stride,
                  const int32_t groupSize,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<ReplicatedAllGatherStridedOp>(
        in,
        out,
        ReplicatedAllGatherStridedOp::defaultOperatorId(),
        uint32_t(stride),
        uint32_t(groupSize),
        settings);
  }

  uint32_t getStride() const { return stride; }
  int64_t getCommSize() const override { return groupSize; }

protected:
  TensorInfo gatheredOutInfo;
  uint32_t stride;
  uint32_t groupSize;
};

const popart::OperatorIdentifier ReplicatedAllGatherStrided =
    ReplicatedAllGatherStridedOp::defaultOperatorId();

// -------------- OpX --------------
namespace popx {

class ReplicatedAllGatherStridedOpx : public CollectivesBaseOpx {
public:
  ReplicatedAllGatherStridedOpx(Op *op, Devicex *devicex)
      : CollectivesBaseOpx(op, devicex) {
    verifyOp<ReplicatedAllGatherStridedOp>(
        op, {ReplicatedAllGatherStridedOp::defaultOperatorId()});
    inputCreatorPriority = -1.0;
  }

  void grow(poplar::program::Sequence &prog) const {
    auto &op = getOp<ReplicatedAllGatherStridedOp>();
    poplar::Tensor toGather =
        getInTensor(ReplicatedAllGatherStridedOp::getInIndex());
    const poplar::OptionFlags &allGatherOptions = dv_p->lowering().gclOptions;

    poplar::Tensor gathered =
        allGatherStrided(graph(),
                         toGather,
                         prog,
                         op.getStride(),
                         op.getCommSize(),
                         debugContext("replicatedAllGatherStrided"),
                         allGatherOptions);

    if (getOp<ReplicatedAllGatherStridedOp>()
            .isConfigureOutputForReplicatedTensorSharding()) {
      auto cbr = getCollectiveBalancedReorder(
          CollectivesBaseOp::getDefaultTensorShardingGroupIndex());
      if (cbr) {
        gathered = cbr->undoRearrangeForCollective(gathered);
      } else {
        throw error("ReplicatedAllGatherOpx::grow, "
                    "CollectiveBalancedReorder not found for Op {}",
                    op_p->debugName());
      }
    }

    setOutTensor(
        ReplicatedAllGatherStridedOp::getOutIndex(),
        gathered.reshape(op.outInfo(ReplicatedAllGatherStridedOp::getOutIndex())
                             .shape_szt()));
  }

  InputCreatorType getInputCreatorType(InIndex index) const {
    return index == ReplicatedAllGatherStridedOp::getInIndex() &&
                   getOp<ReplicatedAllGatherStridedOp>()
                       .isConfigureOutputForReplicatedTensorSharding()
               ? InputCreatorType::CanCreateOrUnwind
               : Opx::getInputCreatorType(index);
  }

  poplar::Tensor
  unwindTensorLayout(poplar::Tensor tensor, InIndex, OutIndex) const {
    auto cbr = createCollectiveBalancedReorder(
        tensor, CollectivesBaseOp::getDefaultTensorShardingGroupIndex());
    return cbr->createReplicaSlice(tensor.elementType());
  }

  view::RegMap unwindRegion(InIndex, OutIndex) const {
    auto info = inInfo(ReplicatedAllGatherStridedOp::getInIndex());
    return [info](const view::Region &) {
      return view::Regions(1, view::Region::getFull(info.shape()));
    };
  }

  std::set<TensorId> mustExistBeforeCreate(InIndex) const { return {}; }

  poplar::Tensor createInputTensor(InIndex index,
                                   const poplar::DebugNameAndId &dnai) const {
    auto &op = getOp<ReplicatedAllGatherStridedOp>();

    if (index == ReplicatedAllGatherStridedOp::getInIndex()) {
      auto outInfo = op.outInfo(ReplicatedAllGatherStridedOp::getOutIndex());
      auto outTensor =
          graph().addVariable(popType(outInfo), outInfo.shape_szt(), dnai);
      dv_p->lowering().getLinearMapper().mapTensor(graph(), outTensor);
      auto cbr = createCollectiveBalancedReorder(
          outTensor, CollectivesBaseOp::getDefaultTensorShardingGroupIndex());
      return cbr->createReplicaSlice(popType(outInfo));
    }

    throw error("createInput: Invalid index = " + std::to_string(index));
  }

  bool hasCreatorViewChangers(InIndex index) const {
    return (index == ReplicatedAllGatherStridedOp::getInIndex());
  }

  ViewChangers getCreatorViewChangers(InIndex index) const {
    if (index == ReplicatedAllGatherStridedOp::getInIndex()) {
      auto group = getCollectiveLinkedGroup(
          CollectivesBaseOp::getDefaultTensorShardingGroupIndex());

      ViewChangers viewChangers(
          {std::make_shared<ReplicatedGatherInScatterOutViewChanger>(
              inInfo(ReplicatedAllGatherStridedOp::getInIndex()).nelms(),
              group.id)});
      return viewChangers;
    }
    throw error("getCreatorViewChangers: Invalid index = " +
                std::to_string(index));
  }
};

popx::OpxCreator<ReplicatedAllGatherStridedOpx>
    ReplicatedAllGatherStridedOpxCreator(ReplicatedAllGatherStrided);

} // namespace popx

} // namespace popart

// -------------- PyBind --------------
// `replicated_all_gather_strided_binding` must equal filename
PYBIND11_MODULE(replicated_all_gather_strided_binding, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<popart::ReplicatedAllGatherStridedOp,
             popart::Op,
             std::shared_ptr<popart::ReplicatedAllGatherStridedOp>>
      binding(m, "ReplicatedAllGatherStridedOp");
  binding.def(py::init<const popart::OperatorIdentifier &,
                       const uint32_t,
                       const uint32_t,
                       const popart::Op::Settings &>(),
              py::arg("opid"),
              py::arg("stride"),
              py::arg("groupSize"),
              py::arg("settings"));
  binding.def_static(
      "createOpInGraph",
      py::overload_cast<popart::Graph &,
                        const InMapType &,
                        const OutMapType &,
                        const int32_t,
                        const int32_t,
                        const popart::Op::Settings &>(
          &popart::ReplicatedAllGatherStridedOp::createOpInGraph),
      py::arg("graph"),
      py::arg("inputs"),
      py::arg("outputs"),
      py::arg("stride"),
      py::arg("groupSize"),
      py::arg("settings"),
      py::return_value_policy::reference);
  binding.def("outTensor",
              py::overload_cast<OutIndex>(
                  &popart::ReplicatedAllGatherStridedOp::outTensor),
              py::return_value_policy::reference);
  binding.def(
      "getGraph",
      py::overload_cast<>(&popart::ReplicatedAllGatherStridedOp::getGraph),
      py::return_value_policy::reference);
};

// -------------- cppimport --------------
// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare']
cfg['sources'] = ['mgcl.cpp']
cfg['libraries'] = ['popart', 'poplar', 'popops']
setup_pybind11(cfg)
%>
*/
