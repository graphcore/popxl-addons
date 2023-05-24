// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// the python module!
// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <popart/attributes.hpp>
#include <popart/datatype.hpp>
#include <popart/logging.hpp>
#include <popart/op.hpp>
#include <popart/op/custom/parameterizedop.hpp>
#include <popart/op/custom/parameterizedopbinder.hpp>
#include <popart/operatoridentifier.hpp>
#include <popart/opserialiser.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>

namespace popart {

// -------------- Op --------------
struct LinearLayoutParams {
  size_t bs;

  void appendAttributes(popart::OpSerialiserBase &os) const {
    os.appendAttribute("batchsize", bs);
  }
};

class LinearLayoutOp
    : public ParameterizedOp<LinearLayoutOp, LinearLayoutParams> {
public:
  using ParameterizedOp<LinearLayoutOp, LinearLayoutParams>::ParameterizedOp;

  static OperatorIdentifier defaultOperatorId() {
    return OperatorIdentifier{"popxl.addons.ops", "LinearLayout", 1, {1, 1}, 1};
  }

  void setup() override {
    auto t = inInfo(0);
    outInfo(0) = TensorInfo(t.data_type(), t.shape());
  }
};

namespace popx {

// -------------- Opx --------------
class LinearLayoutOpx : public Opx {
public:
  LinearLayoutOpx(popart::Op *op, Devicex *devicex) : Opx(op, devicex) {
    verifyOp<LinearLayoutOp>(op, {LinearLayoutOp::defaultOperatorId()});
  }

  void grow(poplar::program::Sequence &pop_prog) const override {
    auto &pop_graph = graph();

    poplar::Tensor input = getInTensor(0);
    auto dataType = input.elementType();
    auto &op = getOp<LinearLayoutOp>();
    auto output = pop_graph.addVariable(dataType, input.shape(), "new_layout");
    poputil::mapTensorLinearly(pop_graph, output, 0, op.params().bs);
    pop_prog.add(poplar::program::Copy(input, output));
    setOutTensor(0, output);
  }
};

namespace {
OpxCreator<LinearLayoutOpx>
    LinearLayoutOpxCreator({LinearLayoutOp::defaultOperatorId()});
}

} // namespace popx

// -------------- PyBind --------------
PYBIND11_MODULE(linear_layout_impl, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<LinearLayoutParams>(m, "LinearLayoutParams")
      .def(py::init<int>(), py::arg("bs"))
      .def_readwrite("bs", &LinearLayoutParams::bs);

  auto cls = popart::ir::op::makeParameterizedOpBindings<LinearLayoutOp>(
      m, "LinearLayoutOp");
}

// pybind end

} // namespace popart

// -------------- cppimport --------------
// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare', '-Wno-attributes', '-Wno-narrowing']
cfg['libraries'] = ['poputil', 'popart']
setup_pybind11(cfg)
%>
*/
