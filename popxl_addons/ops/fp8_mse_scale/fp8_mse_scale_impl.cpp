// cppimport
// NOTE: the cppimport comment is necessary for dynamic compilation when loading
// the python module!
// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <vector>

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

#include <poplar/Graph.hpp>
#include <poplar/MetadataCreation.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/TypeConversion.hpp>
#include <poplar/VariableMappingMethod.hpp>
#include <popnn/Loss.hpp>
#include <popops/AllTrue.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/GatherStatistics.hpp>
#include <popops/Reduce.hpp>
#include <popops/Zero.hpp>

using namespace popops;
namespace pe = popops::expr;
namespace popart {

// -------------- Op --------------
struct Fp8ScalingBiasParams {
  unsigned min_fp32_exp_index;
  unsigned format_span;
  float quantisation_error;

  void appendAttributes(popart::OpSerialiserBase &os) const {
    os.appendAttribute("min_fp32_exp_index", min_fp32_exp_index);
    os.appendAttribute("format_span", format_span);
    os.appendAttribute("quantisation_error", quantisation_error);
  }
};

class Fp8FindBestScaleOp
    : public ParameterizedOp<Fp8FindBestScaleOp, Fp8ScalingBiasParams> {
public:
  using ParameterizedOp<Fp8FindBestScaleOp,
                        Fp8ScalingBiasParams>::ParameterizedOp;

  static OperatorIdentifier defaultOperatorId() {
    return OperatorIdentifier{"popxl.addons.ops", "Fp8Scale", 1, {1, 1}, 1};
  }

  void setup() override { outInfo(0) = TensorInfo(DataType::INT32, Shape{}); }
};

namespace popx {

// -------------- Opx --------------
class Fp8FindBestScaleOpx : public Opx {
public:
  Fp8FindBestScaleOpx(popart::Op *op, Devicex *devicex) : Opx(op, devicex) {
    verifyOp<Fp8FindBestScaleOp>(op, {Fp8FindBestScaleOp::defaultOperatorId()});
  }

  void grow(poplar::program::Sequence &pop_prog) const override {
    auto &pop_graph = graph();

    auto &op = getOp<Fp8FindBestScaleOp>();
    poplar::Tensor src = getInTensor(0).flatten();
    const int min_fp32_exp_index = op.params().min_fp32_exp_index;
    const int format_span = op.params().format_span;
    const float quantisation_error = op.params().quantisation_error;
    // step 1: get histogram, bins, for simplicity, use only fp32 histogram as
    // input.
    auto bins = pop_graph.addVariable(poplar::INT, {254},
                                      poplar::VariableMappingMethod::LINEAR);
    popops::iota(pop_graph, bins, -126, pop_prog);
    bins = popops::cast(pop_graph, bins, poplar::FLOAT, pop_prog);
    popops::mapInPlace(pop_graph, expr::UnaryOpType::EXPONENT2, bins, pop_prog);
    auto hist = popops::histogram(
        pop_graph, src,
        popops::cast(pop_graph, bins, src.elementType(), pop_prog), true,
        pop_prog, debugContext("histogram"));
    hist = popops::cast(pop_graph, hist, poplar::FLOAT, pop_prog);
    auto hist_size = src.numElements();
    popops::mapInPlace(pop_graph, pe::_1 / hist_size, {hist}, pop_prog);

    // step 2: calculate mse
    auto bias = pop_graph.addVariable(poplar::UNSIGNED_INT, {1}, {"bias"});
    pop_graph.setTileMapping(bias, 0);
    auto bias_init = getConst(poplar::UNSIGNED_INT, {1}, 0, "bias_init");
    pop_prog.add(poplar::program::Copy(bias_init, bias));

    auto one = pop_graph.addConstant(poplar::UNSIGNED_INT, {}, 1);
    pop_graph.setTileMapping(one, 0);
    auto fp32_hist_size = pop_graph.addConstant(poplar::UNSIGNED_INT, {}, 254);
    pop_graph.setTileMapping(fp32_hist_size, 0);

    poplar::program::Sequence mse_prog;
    auto mse = pop_graph.addVariable(poplar::FLOAT, {62}, {"mse"});
    pop_graph.setTileMapping(mse, 0);
    popops::zero(pop_graph, mse, pop_prog, debugContext("mse"));

    auto underflow_max =
        pop_graph.addVariable(poplar::UNSIGNED_INT, {}, {"underflow"});
    pop_graph.setTileMapping(underflow_max, 0);

    auto underflow_init =
        getConst(poplar::UNSIGNED_INT, {}, min_fp32_exp_index, "under_init");
    pop_prog.add(poplar::program::Copy(underflow_init, underflow_max));

    auto overflow_min =
        pop_graph.addVariable(poplar::UNSIGNED_INT, {}, {"overflow"});
    pop_graph.setTileMapping(overflow_min, 0);
    auto overflow_init =
        getConst(poplar::UNSIGNED_INT, {}, min_fp32_exp_index + format_span,
                 "over_init");
    pop_prog.add(poplar::program::Copy(overflow_init, overflow_min));

    auto mse_bias = pop_graph.addVariable(poplar::FLOAT, {1}, {"mse_bias"});
    pop_graph.setTileMapping(mse_bias, 0);
    popops::zero(pop_graph, mse_bias, mse_prog, debugContext("mse_bias"));

    auto bin_0 = bins.slice(0, 1, 0);
    hist = hist.slice(1, 255, 0);
    popops::mapInPlace(pop_graph, (pe::_1 * pe::_1 / 4), {bin_0}, pop_prog);

    // calculate mse underflow
    auto counter =
        pop_graph.addVariable(poplar::UNSIGNED_INT, {1}, {"counter"});
    pop_graph.setTileMapping(counter, 0);
    popops::zero(pop_graph, counter, mse_prog, debugContext("counter"));
    addInPlace(pop_graph, counter, one, mse_prog, {"counterIncrement"});

    poplar::program::Sequence cond_under;
    auto underflow = popops::lt(pop_graph, counter, underflow_max, cond_under);
    poplar::program::Sequence underflow_prog;

    auto hist_slice =
        popops::dynamicSlice(pop_graph, hist, counter, {0}, {1}, underflow_prog,
                             debugContext("underflow_slice"));
    auto limits_slice =
        popops::dynamicSlice(pop_graph, bins, counter, {0}, {1}, underflow_prog,
                             debugContext("underflow_slice"));
    popops::mapInPlace(pop_graph, (pe::_1 + pe::_2 * pe::_3 * pe::_3),
                       {mse_bias, hist_slice, limits_slice}, underflow_prog,
                       {"mseUnder"});
    addInPlace(pop_graph, counter, one, underflow_prog, {"counterIncrement"});
    auto predicate =
        popops::allTrue(pop_graph, underflow, cond_under, "underflow");
    mse_prog.add(poplar::program::RepeatWhileTrue(cond_under, predicate,
                                                  underflow_prog));

    // calculate mse winthin range
    poplar::program::Sequence cond_in;
    auto within = popops::gteq(pop_graph, counter, underflow_max, cond_in);
    auto in_right = popops::gt(pop_graph, overflow_min, counter, cond_in);
    popops::logicalAndInPlace(pop_graph, within, in_right, cond_in);
    predicate = popops::allTrue(pop_graph, within, cond_in, "within");

    poplar::program::Sequence within_prog;
    hist_slice =
        popops::dynamicSlice(pop_graph, hist, counter, {0}, {1}, within_prog,
                             debugContext("within_slice"));
    limits_slice =
        popops::dynamicSlice(pop_graph, bins, counter, {0}, {1}, within_prog,
                             debugContext("within_slice"));
    popops::mapInPlace(
        pop_graph, (pe::_1 + pe::_2 * pe::_3 * pe::_3 * quantisation_error),
        {mse_bias, hist_slice, limits_slice}, within_prog, {"mseIn"});
    addInPlace(pop_graph, counter, one, within_prog, {"counterIncrement"});
    mse_prog.add(
        poplar::program::RepeatWhileTrue(cond_in, predicate, within_prog));

    // // calculate mse overflow
    poplar::program::Sequence cond_over;
    auto overflow = popops::gteq(pop_graph, counter, overflow_min, cond_over);
    auto overflow_right =
        popops::lt(pop_graph, counter, fp32_hist_size, cond_over);
    popops::logicalAndInPlace(pop_graph, overflow, overflow_right, cond_over);
    predicate = popops::allTrue(pop_graph, overflow, cond_over, "over");

    poplar::program::Sequence over_prog;
    auto over_limit = limits_slice;
    hist_slice = popops::dynamicSlice(pop_graph, hist, counter, {0}, {1},
                                      over_prog, debugContext("over_slice"));
    limits_slice = popops::dynamicSlice(pop_graph, bins, counter, {0}, {1},
                                        over_prog, debugContext("over_slice"));
    popops::mapInPlace(
        pop_graph, (pe::_1 + pe::_2 * (pe::_3 - pe::_4) * (pe::_3 - pe::_4)),
        {mse_bias, hist_slice, over_limit, limits_slice}, over_prog,
        {"mseOver"});
    addInPlace(pop_graph, counter, one, over_prog, {"counterIncrement"});
    mse_prog.add(
        poplar::program::RepeatWhileTrue(cond_over, predicate, over_prog));
    pop_prog.add(poplar::program::Copy(overflow_init, overflow_min));
    // Update the mses for all biases
    popops::dynamicUpdate(pop_graph, mse, mse_bias, bias, {0}, {1}, mse_prog,
                          {"updateMSE"});
    addInPlace(pop_graph, bias, one, mse_prog, {"biasIncrement"});
    addInPlace(pop_graph, underflow_max, one, mse_prog, {"biasIncrement"});
    addInPlace(pop_graph, overflow_min, one, mse_prog, {"biasIncrement"});
    pop_prog.add(poplar::program::Repeat(62, mse_prog, "mse"));
    pop_prog.add(poplar::program::PrintTensor("final mse", mse));
    auto min_bias =
        popnn::minAndArgMin(pop_graph, mse.expand({0}), pop_prog, {"argmin"});

    popops::mapInPlace(pop_graph, pe::Cast((pe::_1 == pe::_2), poplar::INT),
                       {mse, min_bias.first}, pop_prog, {"min_mse"});
    auto sum = popops::reduce(pop_graph, mse, {0}, {popops::Operation::ADD},
                              pop_prog, debugContext("delta"));
    pop_prog.add(poplar::program::PrintTensor("nb of min mse", sum));

    pop_prog.add(poplar::program::PrintTensor("bias is", bias));
    auto scaling_bias = popops::cast(pop_graph, min_bias.second.squeeze({0}),
                                     poplar::INT, pop_prog);
    popops::mapInPlace(pop_graph,
                       -(pe::_1 + pe::Cast(pe::_2 / 2, poplar::INT) - 31),
                       {scaling_bias, sum}, pop_prog, {"scaling bias"});

    // pop_prog.add(poplar::program::PrintTensor("scaling bias is",
    // scaling_bias));
    setOutTensor(0, scaling_bias);
  }
};

namespace {
OpxCreator<Fp8FindBestScaleOpx>
    Fp8FindBestScaleOpxCreator({Fp8FindBestScaleOp::defaultOperatorId()});
}

} // namespace popx

// -------------- PyBind --------------
PYBIND11_MODULE(fp8_mse_scale_impl, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<Fp8ScalingBiasParams>(m, "Fp8ScalingBiasParams")
      .def(py::init<int, int, float>(), py::arg("min_fp32_exp_index"),
           py::arg("format_span"), py::arg("quantisation_error"))
      .def_readwrite("min_fp32_exp_index",
                     &Fp8ScalingBiasParams::min_fp32_exp_index)
      .def_readwrite("format_span", &Fp8ScalingBiasParams::format_span)
      .def_readwrite("quantisation_error",
                     &Fp8ScalingBiasParams::quantisation_error);

  auto cls = popart::ir::op::makeParameterizedOpBindings<Fp8FindBestScaleOp>(
      m, "Fp8FindBestScaleOp");
}
// pybind end

} // namespace popart

// -------------- cppimport --------------
// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare', '-Wno-attributes', '-Wno-narrowing']
cfg['libraries'] = ['popart']
setup_pybind11(cfg)
%>
*/
