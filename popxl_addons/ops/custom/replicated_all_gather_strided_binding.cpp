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
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>
#include <snap/Tensor.hpp>
#include <vector>

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

using InMapType = std::map<popart::InIndex, popart::TensorId>;
using OutMapType = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex = int;

namespace popart {

#define CUSTOM_OP_DOMAIN "custom.ops"

// -------------- Op --------------
class ReplicatedAllGatherStridedOp : public CollectivesBaseOp {
public:
  ReplicatedAllGatherStridedOp(const OperatorIdentifier &_opid,
                               const uint32_t stride_,
                               const uint32_t groupSize_,
                               const Op::Settings &settings_)
      : CollectivesBaseOp(
            _opid,
            stride_ == 1 ? CommGroup(CommGroupType::Consecutive, groupSize_)
                         : CommGroup(CommGroupType::None, 0),
            settings_),
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

    logging::op::trace("[ReplicatedAllGatherOp] Global replication factor: {}, "
                       "sharding factor: {}, stride {}, group size {}",
                       globalReplicationFactor, replicationFactor, stride,
                       groupSize);
  }

  float getSubgraphValue() const final { return getHighSubgraphValue(); }

  void appendOutlineAttributes(OpSerialiserBase &os) const {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("stride", stride);
    os.appendAttribute("groupSize", groupSize);
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
  createOpInGraph(popart::Graph &graph, const InMapType &in,
                  const OutMapType &out, const int32_t stride,
                  const int32_t groupSize,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<ReplicatedAllGatherStridedOp>(
        in, out, ReplicatedAllGatherStridedOp::defaultOperatorId(),
        uint32_t(stride), uint32_t(groupSize), settings);
  }

  uint32_t getStride() const { return stride; }
  uint32_t getGroupSize() const { return groupSize; }

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
  }

  void grow(snap::program::Sequence &prog) const {
    auto &op = getOp<ReplicatedAllGatherStridedOp>();

    const poplar::OptionFlags &allGatherOptions = dv_p->lowering().gclOptions;
    poplar::Tensor gathered;
    uint32_t stride = op.getStride(), groupSize = op.getGroupSize();
    if (stride == 1) {
      gathered = gcl::allGatherCrossReplica(
          graph().getPoplarGraph(),
          getInTensor(ReplicatedAllGatherOp::getInIndex()).getPoplarTensor(),
          prog.getPoplarSequence(),
          toGCLCommGroup(popart::CommGroup(popart::CommGroupType::Consecutive,
                                           op.getGroupSize())),
          debugContext("replicatedAllGatherStrided"), allGatherOptions);
    } else {
      if (stride == 64) {
        gathered = gcl::allGatherCrossReplica(
            graph().getPoplarGraph(),
            getInTensor(ReplicatedAllGatherOp::getInIndex()).getPoplarTensor(),
            prog.getPoplarSequence(),
            toGCLCommGroup(popart::CommGroup(popart::CommGroupType::Orthogonal,
                                             op.getGroupSize())),
            debugContext("replicatedAllGatherStrided"), allGatherOptions);
      } else {
        if (stride * groupSize == 64) {
          gathered =
              ringAllGather(graph().getPoplarGraph(),
                            getInTensor(ReplicatedAllGatherOp::getInIndex())
                                .getPoplarTensor(),

                            prog.getPoplarSequence(), stride, groupSize);
        } else {
          gathered =
              maskedAllGather(graph().getPoplarGraph(),
                              getInTensor(ReplicatedAllGatherOp::getInIndex())
                                  .getPoplarTensor(),
                              prog.getPoplarSequence(), stride, groupSize);
        }
      }
    }

    if (hasInput(ReplicatedAllGatherStridedOp::getCollectiveLinkedIndex())) {
      auto cbr = getCollectiveBalancedReorder(
          CollectivesBaseOp::getDefaultTensorShardingGroupIndex());
      if (cbr) {
        gathered = cbr->undoRearrangeForCollective(gathered);
      } else {
        throw error("ReplicatedAllGatherStridedOpx::grow, "
                    "CollectiveBalancedReorder not found for Op {}",
                    op_p->debugName());
      }
    }

    setOutTensor(
        ReplicatedAllGatherStridedOp::getOutIndex(),
        snap::Tensor{gathered.reshape(
                         op.outInfo(ReplicatedAllGatherStridedOp::getOutIndex())
                             .shape_szt()),
                     graph()});
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
  py::class_<popart::ReplicatedAllGatherStridedOp, popart::Op,
             std::shared_ptr<popart::ReplicatedAllGatherStridedOp>>
      binding(m, "ReplicatedAllGatherStridedOp");
  binding.def(py::init<const popart::OperatorIdentifier &, const uint32_t,
                       const uint32_t, const popart::Op::Settings &>(),
              py::arg("opid"), py::arg("stride"), py::arg("groupSize"),
              py::arg("settings"));
  binding.def_static(
      "createOpInGraph",
      py::overload_cast<popart::Graph &, const InMapType &, const OutMapType &,
                        const int32_t, const int32_t,
                        const popart::Op::Settings &>(
          &popart::ReplicatedAllGatherStridedOp::createOpInGraph),
      py::arg("graph"), py::arg("inputs"), py::arg("outputs"),
      py::arg("stride"), py::arg("groupSize"), py::arg("settings"),
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
cfg['libraries'] = ['popart']
setup_pybind11(cfg)
%>
*/
