// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// Copyright (c) 2022 Graphcore Ltd. All rights reserved.

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
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/region.hpp>
#include <popart/tensor.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>
#include <poplar/Tensor.hpp>
#include <vector>

#include <popart/op/collectives/collectives.hpp>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "layer_norm_distributed.hpp"
#include "layer_norm_distributedx.hpp"

namespace py = pybind11;

// -------------- PyBind --------------
// `layer_norm_distributed_binding` must equal filename
PYBIND11_MODULE(layer_norm_distributed_binding, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<popart::LayerNormDistributedOp, popart::Op,
             std::shared_ptr<popart::LayerNormDistributedOp>>
      binding(m, "LayerNormDistributedOp");
  binding.def_static(
      "createOpInGraph",
      py::overload_cast<popart::Graph &, const InMapType &, const OutMapType &,
                        float, const popart::ReplicaGrouping &,
                        const popart::Op::Settings &>(
          &popart::LayerNormDistributedOp::createOpInGraph),
      py::arg("graph"), py::arg("inputs"), py::arg("outputs"),
      py::arg("epsilon"), py::arg("group"), py::arg("settings"),
      py::return_value_policy::reference);
  binding.def(
      "outTensor",
      py::overload_cast<OutIndex>(&popart::LayerNormDistributedOp::outTensor),
      py::return_value_policy::reference);
};

// -------------- cppimport --------------
// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['sources'] = ['layer_norm_distributed.cpp', 'layer_norm_distributedx.cpp']
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare']
cfg['libraries'] = ['popart', 'poputil', 'popops', 'poplin', 'popnn', 'poprand', 'gcl']
setup_pybind11(cfg)
%>
*/
