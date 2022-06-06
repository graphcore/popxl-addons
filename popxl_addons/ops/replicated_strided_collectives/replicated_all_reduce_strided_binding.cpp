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
#include <snap/Tensor.hpp>
#include <vector>
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

#include <popart/op/collectives/collectives.hpp>
#include <popart/op/collectives/replicatedallreduce.hpp>
#include <popart/popx/op/collectives/replicatedallreducex.hpp>

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
class ReplicatedAllReduceStridedOp : public ReplicatedAllReduceOp {
public:
  ReplicatedAllReduceStridedOp(const OperatorIdentifier &_opid,
                               const CollectiveOperator &op_,
                               const uint32_t stride_,
                               const uint32_t groupSize_,
                               const bool identicalInputs_,
                               const bool identicalGradInputs_,
                               const Op::Settings &settings_)
      : ReplicatedAllReduceOp(
            _opid,
            op_,
            stride_ == 1 ? CommGroup(CommGroupType::Consecutive, groupSize_)
                         : CommGroup(CommGroupType::None, 0),
            settings_),
        stride(stride_), groupSize(groupSize_),
        identicalInputs(identicalInputs_),
        identicalGradInputs(identicalGradInputs_) {}

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<ReplicatedAllReduceStridedOp>(*this);
  }

  void setup() override {
    if (op != CollectiveOperator::Add) {
      throw error("Cannot create ReplicatedAllReduceTPOp op. "
                  "CollectiveOperator::Add is the only collective operator "
                  "that is currently implemented.");
    }
    if (stride == 0 || groupSize == 0) {
      throw error("Cannot create ReplicatedAllGatherStrided op "
                  "stride and group size must be > 0.");
    }
    ReplicatedAllReduceOp::setup();
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> result;
    // Reverse identicalInputs <-> identicalGradInputs
    result.push_back(
        std::make_unique<ReplicatedAllReduceStridedOp>(opid,
                                                       op,
                                                       stride,
                                                       groupSize,
                                                       identicalGradInputs,
                                                       identicalInputs,
                                                       settings));
    return result;
  }

  void appendOutlineAttributes(OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("op", op);
    os.appendAttribute("stride", stride);
    os.appendAttribute("groupSize", groupSize);
    os.appendAttribute("identicalInputs", identicalInputs);
    os.appendAttribute("identicalGradInputs", identicalGradInputs);
  }

  // Replace the op with an idenity if identicalInputs==true
  bool canBeReplacedByIdentity() const override { return identicalInputs; }

  const std::vector<GradInOutMapper> &gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
        {getInIndex(), getOutIndex(), GradOpInType::GradOut}};
    return inInfo;
  }

  const std::map<int, int> &gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {{getOutIndex(), getInIndex()}};
    return outInfo;
  }

  static OperatorIdentifier defaultOperatorId() {
    return OperatorIdentifier{
        CUSTOM_OP_DOMAIN,
        "ReplicatedAllReduceStrided",
        1,      // Op version
        {1, 1}, // number of inputs
        1       // number of outputs
    };
  }

  static ReplicatedAllReduceStridedOp *
  createOpInGraph(popart::Graph &graph,
                  const InMapType &in,
                  const OutMapType &out,
                  const CollectiveOperator &op,
                  const int32_t stride,
                  const int32_t groupSize,
                  const bool identicalInputs,
                  const bool identicalGradInputs,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<ReplicatedAllReduceStridedOp>(
        in,
        out,
        ReplicatedAllReduceStridedOp::defaultOperatorId(),
        op,
        uint32_t(stride),
        uint32_t(groupSize),
        identicalInputs,
        identicalGradInputs,
        settings);
  }

  uint32_t getStride() const { return stride; }
  uint32_t getGroupSize() const { return groupSize; }

protected:
  uint32_t stride;
  uint32_t groupSize;
  bool identicalInputs     = false;
  bool identicalGradInputs = false;
};

const popart::OperatorIdentifier ReplicatedAllReduceStrided =
    ReplicatedAllReduceStridedOp::defaultOperatorId();

// -------------- OpX --------------
namespace popx {

class ReplicatedAllReduceStridedOpx : public ReplicatedAllReduceOpx {
public:
  ReplicatedAllReduceStridedOpx(Op *op, Devicex *devicex)
      : ReplicatedAllReduceOpx(op, devicex) {
    verifyOp<ReplicatedAllReduceStridedOp>(
        op, {ReplicatedAllReduceStridedOp::defaultOperatorId()});
  }

  void grow(snap::program::Sequence &prog) const {
    const auto &rarOp = getOp<ReplicatedAllReduceStridedOp>();

    const auto inIndex      = ReplicatedAllReduceOp::getInIndex();
    poplar::Tensor toReduce = getInTensor(inIndex).getPoplarTensor();
    const poplar::OptionFlags &allReduceOptions = dv_p->lowering().gclOptions;
    poplar::Tensor output;
    uint32_t stride = rarOp.getStride(), groupSize = rarOp.getGroupSize();
    if (stride == 1) {
      output = gcl::allReduceCrossReplica(
          graph().getPoplarGraph(),
          toReduce,
          getPoplarCollectiveOperator(rarOp.getCollectiveOp()),
          prog.getPoplarSequence(),
          toGCLCommGroup(popart::CommGroup(popart::CommGroupType::Consecutive,
                                           rarOp.getGroupSize())),
          debugContext("replicatedAllReduceStrided"),
          allReduceOptions);
    } else {
      if (stride == 64) {
        output = gcl::allReduceCrossReplica(
            graph().getPoplarGraph(),
            toReduce,
            getPoplarCollectiveOperator(rarOp.getCollectiveOp()),
            prog.getPoplarSequence(),
            toGCLCommGroup(popart::CommGroup(popart::CommGroupType::Orthogonal,
                                             rarOp.getGroupSize())),
            debugContext("replicatedAllReduceStrided"),
            allReduceOptions);
      } else {
        if (stride * groupSize == 64) {
          output = ringAllReduce(graph().getPoplarGraph(),
                                 toReduce,
                                 prog.getPoplarSequence(),
                                 stride,
                                 groupSize);
        } else {
          output = maskedAllReduce(graph().getPoplarGraph(),
                                   toReduce,
                                   prog.getPoplarSequence(),
                                   stride,
                                   groupSize);
        }
      }
    }
    logging::transform::trace(
        "[ReplicatedAllReduceStridedOpx::grow] stride: {}, groupSize {},"
        "input shape: {}, output shape: {}",
        rarOp.getStride(),
        rarOp.getGroupSize(),
        toReduce.shape(),
        output.shape());

    if (hasInViewChangers(ReplicatedAllReduceOp::getInIndex())) {
      setOutViewChangers(
          ReplicatedAllReduceOp::getOutIndex(),
          getInViewChangers(ReplicatedAllReduceOp::getInIndex()));
    }
    setOutTensor(ReplicatedAllReduceOp::getOutIndex(),
                 snap::Tensor{output, graph()});
  }
};

popx::OpxCreator<ReplicatedAllReduceStridedOpx>
    ReplicatedAllReduceStridedOpxCreator(ReplicatedAllReduceStrided);

} // namespace popx

} // namespace popart

// -------------- PyBind --------------
// `replicated_all_reduce_strided_binding` must equal filename
PYBIND11_MODULE(replicated_all_reduce_strided_binding, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<popart::ReplicatedAllReduceStridedOp,
             popart::Op,
             std::shared_ptr<popart::ReplicatedAllReduceStridedOp>>
      binding(m, "ReplicatedAllReduceStridedOp");
  binding.def(py::init<const popart::OperatorIdentifier &,
                       const popart::CollectiveOperator &,
                       const uint32_t,
                       const uint32_t,
                       const bool,
                       const bool,
                       const popart::Op::Settings &>(),
              py::arg("opid"),
              py::arg("op"),
              py::arg("stride"),
              py::arg("groupSize"),
              py::arg("identicalInputs"),
              py::arg("identicalGradInputs"),
              py::arg("settings"));
  binding.def_static(
      "createOpInGraph",
      py::overload_cast<popart::Graph &,
                        const InMapType &,
                        const OutMapType &,
                        const popart::CollectiveOperator &,
                        const int32_t,
                        const int32_t,
                        const bool,
                        const bool,
                        const popart::Op::Settings &>(
          &popart::ReplicatedAllReduceStridedOp::createOpInGraph),
      py::arg("graph"),
      py::arg("inputs"),
      py::arg("outputs"),
      py::arg("op"),
      py::arg("stride"),
      py::arg("groupSize"),
      py::arg("identicalInputs"),
      py::arg("identicalGradInputs"),
      py::arg("settings"),
      py::return_value_policy::reference);
  binding.def("outTensor",
              py::overload_cast<OutIndex>(
                  &popart::ReplicatedAllReduceStridedOp::outTensor),
              py::return_value_policy::reference);
  binding.def(
      "getGraph",
      py::overload_cast<>(&popart::ReplicatedAllReduceStridedOp::getGraph),
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