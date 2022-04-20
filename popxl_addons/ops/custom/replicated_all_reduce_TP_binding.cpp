// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

/*
For the op to work you need to run the `OpToIdentityPattern` which is called as part of `PreAliasPatterns`
after autodiffing the op.

You also need python header files:
sudo apt install libpython3.6-dev
*/

#include <memory>
#include <vector>
#include <map>
#include <snap/Tensor.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/op.hpp>
#include <popart/ir.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/error.hpp>
#include <popart/util.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/popx/popopx.hpp>
#include <popart/basicoptionals.hpp>
#include <popart/graph.hpp>

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

namespace py = pybind11;

using InMapType     = std::map<popart::InIndex, popart::TensorId>;
using OutMapType    = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex = int;

namespace popart {

#define CUSTOM_OP_DOMAIN "popxl.addons.ops.custom"


// -------------- Op --------------
class ReplicatedAllReduceTPOp : public ReplicatedAllReduceOp {
public:

  ReplicatedAllReduceTPOp(
    const OperatorIdentifier &_opid,
    const CollectiveOperator &op_,
    const CommGroup &group_,
    const bool identicalInputs_,
    const bool identicalGradInputs_,
    const Op::Settings &settings_)
    : ReplicatedAllReduceOp(_opid, op_, group_, settings_),
      identicalInputs(identicalInputs_),
      identicalGradInputs(identicalGradInputs_) {}

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<ReplicatedAllReduceTPOp>(*this);
  }

  void setup() override {
    if (op != CollectiveOperator::Add) {
      throw error("Cannot create ReplicatedAllReduceTPOp op. "
                  "CollectiveOperator::Add is the only collective operator "
                  "that is currently implemented.");
    }
    ReplicatedAllReduceOp::setup();
  }

  std::vector<std::unique_ptr<Op>> getGradOps() override {
    std::vector<std::unique_ptr<Op>> result;
    // Reverse identicalInputs <-> identicalGradInputs
    result.push_back(std::make_unique<ReplicatedAllReduceTPOp>(
        opid, op, getGCLCommGroup(), identicalGradInputs, identicalInputs, settings));
    return result;
  }

  void appendOutlineAttributes(OpSerialiserBase &os) const override {
    Op::appendOutlineAttributes(os);
    os.appendAttribute("op", op);
    os.appendAttribute("group", getGCLCommGroup());
    os.appendAttribute("identicalInputs", identicalInputs);
    os.appendAttribute("identicalGradInputs", identicalGradInputs);
  }

  // Replace the op with an idenity if identicalInputs==true
  bool canBeReplacedByIdentity() const override {
    return identicalInputs;
  }

  const std::vector<GradInOutMapper>& gradInputInfo() const override {
    static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), getOutIndex(), GradOpInType::GradOut}};
    return inInfo;
  }

  const std::map<int, int>& gradOutToNonGradIn() const override {
    static const std::map<int, int> outInfo = {
      {getOutIndex(), getInIndex()}};
    return outInfo;
  } 

  static OperatorIdentifier defaultOperatorId() {
    return OperatorIdentifier{
          CUSTOM_OP_DOMAIN,
          "ReplicatedAllReduceTP",
          1, // Op version
          {1, 1}, // number of inputs
          1 // number of outputs
    };
  }


  static ReplicatedAllReduceTPOp *
  createOpInGraph(popart::Graph &graph,
                  const InMapType &in,
                  const OutMapType &out,
                  const CollectiveOperator &op,
                  const CommGroup &group,
                  const bool identicalInputs,
                  const bool identicalGradInputs,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<ReplicatedAllReduceTPOp>(
        in,
        out,
        ReplicatedAllReduceTPOp::defaultOperatorId(),
        op,
        group,
        identicalInputs,
        identicalGradInputs,
        settings);
  }


protected:
  bool identicalInputs = false;
  bool identicalGradInputs = false;

};

const popart::OperatorIdentifier ReplicatedAllReduceTP = ReplicatedAllReduceTPOp::defaultOperatorId();

// -------------- OpX --------------
namespace popx {

class ReplicatedAllReduceTPOpx : public ReplicatedAllReduceOpx {
public:
  ReplicatedAllReduceTPOpx(Op *op, Devicex *devicex): ReplicatedAllReduceOpx(op, devicex) {
    verifyOp<ReplicatedAllReduceTPOp>(op, {ReplicatedAllReduceTPOp::defaultOperatorId()});
  }

};

popx::OpxCreator<ReplicatedAllReduceTPOpx> ReplicatedAllReduceTPOpxCreator(ReplicatedAllReduceTP);

} // namespace popx

} // namespace popart

// -------------- PyBind --------------
// `replicated_all_reduce_TP_binding` must equal filename
PYBIND11_MODULE(replicated_all_reduce_TP_binding, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<popart::ReplicatedAllReduceTPOp, popart::Op, std::shared_ptr<popart::ReplicatedAllReduceTPOp>> binding(m, "ReplicatedAllReduceTPOp");
  binding.def(py::init<const popart::OperatorIdentifier &,
                  const popart::CollectiveOperator &,
                  const popart::CommGroup &,
                  const bool,
                  const bool,
                  const popart::Op::Settings &>(),
              py::arg("opid"),
              py::arg("op"),
              py::arg("group"),
              py::arg("identicalInputs"),
              py::arg("identicalGradInputs"),
              py::arg("settings"));
  binding.def_static("createOpInGraph",
              py::overload_cast<popart::Graph &,
                                const InMapType &,
                                const OutMapType &,
                                const popart::CollectiveOperator &,
                                const popart::CommGroup &,
                                const bool,
                                const bool,
                                const popart::Op::Settings &>(
                  &popart::ReplicatedAllReduceTPOp::createOpInGraph),
              py::arg("graph"),
              py::arg("inputs"),
              py::arg("outputs"),
              py::arg("op"),
              py::arg("group"),
              py::arg("identicalInputs"),
              py::arg("identicalGradInputs"),
              py::arg("settings"),
              py::return_value_policy::reference);
  binding.def("outTensor",
           py::overload_cast<OutIndex>(&popart::ReplicatedAllReduceTPOp::outTensor),
           py::return_value_policy::reference);
  binding.def("getGraph",
           py::overload_cast<>(&popart::ReplicatedAllReduceTPOp::getGraph),
           py::return_value_policy::reference);
};

// -------------- cppimport --------------
// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare']
cfg['libraries'] = ['popart']
setup_pybind11(cfg)
%>
*/
