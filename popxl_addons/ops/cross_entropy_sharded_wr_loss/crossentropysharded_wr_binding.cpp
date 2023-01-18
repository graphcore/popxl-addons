// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

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

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "crossentropysharded_wr.hpp"
#include "crossentropyshardedx_wr.hpp"

namespace py = pybind11;

// -------------- PyBind --------------
// `crossentropysharded_wr_binding` must equal filename
PYBIND11_MODULE(crossentropysharded_wr_binding, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<popart::CrossEntropyShardedWROp,
             popart::Op,
             std::shared_ptr<popart::CrossEntropyShardedWROp>>
      binding(m, "CrossEntropyShardedOp");
  binding.def_static("createOpInGraph",
                     py::overload_cast<popart::Graph &,
                                       const InMapType &,
                                       const OutMapType &,
                                       const std::vector<int64_t>,
                                       const popart::Op::Settings &>(
                         &popart::CrossEntropyShardedWROp::createOpInGraph),
                     py::arg("graph"),
                     py::arg("inputs"),
                     py::arg("outputs"),
                     py::arg("ipus"),
                     py::arg("settings"),
                     py::return_value_policy::reference);
  binding.def(
      "outTensor",
      py::overload_cast<OutIndex>(&popart::CrossEntropyShardedWROp::outTensor),
      py::return_value_policy::reference);
};

// -------------- cppimport --------------
// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['sources'] = ['crossentropysharded_wr.cpp', 'crossentropyshardedx_wr.cpp']
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare']
cfg['libraries'] = ['popart']
setup_pybind11(cfg)
%>
*/
