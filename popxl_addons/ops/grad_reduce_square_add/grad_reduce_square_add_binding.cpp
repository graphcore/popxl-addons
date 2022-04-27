// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

#include <map>
#include <memory>
#include <snap/Tensor.hpp>
#include <vector>
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

#include <popops/Reduce.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using InMapType  = std::map<popart::InIndex, popart::TensorId>;
using OutMapType = std::map<popart::OutIndex, popart::TensorId>;
using OutIndex   = int;

namespace popart {

#define CUSTOM_OP_DOMAIN "popxl.addons.ops.custom"

// -------------- Op --------------
class GradReduceSquareAddOp : public Op {
public:
  GradReduceSquareAddOp(const Op::Settings &settings_)
      : Op(GradReduceSquareAddOp::defaultOperatorId(), settings_) {}

  std::unique_ptr<Op> clone() const override {
    return std::make_unique<GradReduceSquareAddOp>(*this);
  }

  void setup() override { outInfo(0) = TensorInfo(DataType::FLOAT, Shape{}); }

  static OperatorIdentifier defaultOperatorId() {
    return OperatorIdentifier{
        CUSTOM_OP_DOMAIN,
        "GradReduceSquareAdd",
        1,      // Op version
        {2, 2}, // number of inputs
        1       // number of outputs
    };
  }

  float getSubgraphValue() const override { return getHighSubgraphValue(); }

  static GradReduceSquareAddOp *
  createOpInGraph(popart::Graph &graph,
                  const InMapType &in,
                  const OutMapType &out,
                  const popart::Op::Settings &settings) {
    return graph.createConnectedOp<GradReduceSquareAddOp>(in, out, settings);
  }
};

// -------------- OpX --------------
namespace popx {

class GradReduceSquareAddOpx : public PopOpx {
public:
  GradReduceSquareAddOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
    verifyOp<GradReduceSquareAddOp>(
        op, {GradReduceSquareAddOp::defaultOperatorId()});
  }

  void grow(snap::program::Sequence &prog) const {
    auto to_reduce = getInTensor(0).flatten().getPoplarTensor();
    auto scale     = getInTensor(1).getPoplarTensor();
    auto rsq       = popops::reduce(graph().getPoplarGraph(),
                              to_reduce,
                              poplar::FLOAT,
                              {0},
                              {popops::Operation::SQUARE_ADD, false, scale},
                              prog.getPoplarSequence(),
                              debugContext("GradReduceSquareAddOpx"));

    setOutTensor(0, snap::Tensor{rsq, graph()});
  }
};

popx::OpxCreator<GradReduceSquareAddOpx>
    GradReduceSquareAddOpxCreator(GradReduceSquareAddOp::defaultOperatorId());

} // namespace popx

} // namespace popart

// -------------- PyBind --------------
// `grad_reduce_square_add_binding` must equal filename
PYBIND11_MODULE(grad_reduce_square_add_binding, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<popart::GradReduceSquareAddOp,
             popart::Op,
             std::shared_ptr<popart::GradReduceSquareAddOp>>
      binding(m, "GradReduceSquareAddOp");
  binding.def(py::init<const popart::Op::Settings &>(), py::arg("settings"));
  binding.def_static("createOpInGraph",
                     py::overload_cast<popart::Graph &,
                                       const InMapType &,
                                       const OutMapType &,
                                       const popart::Op::Settings &>(
                         &popart::GradReduceSquareAddOp::createOpInGraph),
                     py::arg("graph"),
                     py::arg("inputs"),
                     py::arg("outputs"),
                     py::arg("settings"),
                     py::return_value_policy::reference);
  binding.def(
      "outTensor",
      py::overload_cast<OutIndex>(&popart::GradReduceSquareAddOp::outTensor),
      py::return_value_policy::reference);
  binding.def("getGraph",
              py::overload_cast<>(&popart::GradReduceSquareAddOp::getGraph),
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
