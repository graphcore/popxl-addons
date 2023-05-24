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
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

#include <popsparse/MatMul.hpp>
#include <popsparse/MatMulParams.hpp>
#include <popsparse/SparsePartitioner.hpp>
#include <popsparse/SparseStorageFormats.hpp>
#include <popsparse/SparseTensor.hpp>
#include <popsparse/codelets.hpp>

namespace {
// cast indices to size_t that is accepted in popsparse
std::vector<size_t> to_szt(const std::vector<int64_t> &from) {
  std::vector<size_t> szts;
  szts.reserve(from.size());
  for (auto &x : from) {
    szts.push_back(static_cast<size_t>(x));
  }
  return szts;
}

// Calculate hash for indices
std::size_t hash_index(std::vector<int64_t> const &vec) {
  std::size_t seed = vec.size();
  for (auto x : vec) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

// Calculate hash for non-zero values
std::size_t hash_nz(std::vector<float> const &vec) {
  std::size_t seed = vec.size();
  std::hash<float> hash_float;
  for (auto x : vec) {
    x = hash_float(x);
    seed ^= (size_t)x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

} // namespace

namespace popart {

// -------------- Op --------------
struct DenseSparseMatMulParams {
  std::vector<float> nz;
  std::vector<int64_t> cols;
  std::vector<int64_t> rows;
  unsigned output_cols;
  nonstd::optional<float> available_memory_proportion;
  // Compute the hash of CSR representation to speed up the `appendAttribute`
  // since it will be called multiple times during lowering.
  size_t nz_hash = hash_nz(nz);
  size_t cols_hash = hash_index(cols);
  size_t rows_hash = hash_index(rows);

  void appendAttributes(popart::OpSerialiserBase &os) const {
    os.appendAttribute("nz", nz_hash);
    os.appendAttribute("cols", cols_hash);
    os.appendAttribute("rows", rows_hash);

    os.appendAttribute("output_cols", output_cols);
    os.appendAttribute(sAvailMemAttribute, available_memory_proportion);
  }
};

class DenseSparseMatMulOp
    : public ParameterizedOp<DenseSparseMatMulOp, DenseSparseMatMulParams> {
public:
  using ParameterizedOp<DenseSparseMatMulOp,
                        DenseSparseMatMulParams>::ParameterizedOp;

  static OperatorIdentifier defaultOperatorId() {
    return OperatorIdentifier{"popxl.addons.ops", "SparseMatmul", 1, {1, 1}, 1};
  }

  bool isOutlineable() const override { return false; }

  void setAvailableMemoryProportion(const nonstd::optional<float> v) {
    m_params.available_memory_proportion = v;
  }

  void setup() override {
    auto lhs = inInfo(0);
    auto output_shape = lhs.shape();
    // Note: ignoring groups
    output_shape[output_shape.size() - 1] = params().output_cols;
    outInfo(0) = TensorInfo(lhs.data_type(), output_shape);
  }
};

namespace popx {

// -------------- Opx --------------
class DenseSparseMatMulOpx : public Opx {
public:
  DenseSparseMatMulOpx(popart::Op *op, Devicex *devicex) : Opx(op, devicex) {
    verifyOp<DenseSparseMatMulOp>(op,
                                  {DenseSparseMatMulOp::defaultOperatorId()});
  }

  void grow(poplar::program::Sequence &pop_prog) const override {
    using namespace popsparse;

    auto &pop_graph = graph();
    popsparse::addCodelets(pop_graph);

    auto &op = getOp<DenseSparseMatMulOp>();
    poplar::Tensor dense_lhs = getInTensor(0);

    // Add group dim if needed
    if (dense_lhs.rank() == 2) {
      dense_lhs = dense_lhs.expand({0});
    }

    auto lhs_shape = dense_lhs.shape();
    auto lhs_dtype = dense_lhs.elementType();
    size_t groups = lhs_shape[0];
    size_t m = lhs_shape[1];
    size_t k = lhs_shape[2];
    size_t n = op.params().output_cols;

    auto rhs_csr =
        CSRMatrix<float>(k, n, op.params().nz, to_szt(op.params().cols),
                         to_szt(op.params().rows), {1, 1}); // block size is 1
    auto params = static_::MatMulParams::createForDenseSparse(groups, m, k, n);

    poplar::OptionFlags opts;
    static_::PlanningCache cache;

    if (op.params().available_memory_proportion) {
      opts.set("availableMemoryProportion",
               std::to_string(*op.params().available_memory_proportion));
    }
    // Create sparse tensor with empty NZ values.
    // Its layout on device is also handled.
    auto sparse_rhs = static_::createDenseSparseMatMulRHS(
        pop_graph, lhs_dtype, params, rhs_csr, debugContext("rhs"), opts,
        &cache);
    auto nz_tensor = sparse_rhs.getNzValuesTensor();

    // Copy the NZ values in the CSR matrix from host to the empty NZ tensor on
    // device The `partitioner` object is first created for the host side data
    // manipulation. Method `createSparsityDataImpl` is then called to create a
    // host representation of sparsity used by the device implementation.
    auto device_data_impl =
        static_::Partitioner<float>(params, lhs_dtype, pop_graph.getTarget(),
                                    opts, &cache)
            .createSparsityDataImpl(rhs_csr);
    auto &device_nz_values = device_data_impl.nzValues;

    if (nz_tensor.elementType() == poplar::HALF) {
      std::vector<uint16_t> nz_half(device_nz_values.size());
      poplar::copyFloatToDeviceHalf(pop_graph.getTarget(),
                                    device_nz_values.data(), nz_half.data(),
                                    nz_half.size());
      pop_graph.setInitialValueHalf(
          nz_tensor,
          poplar::ArrayRef<uint16_t>(nz_half.data(), nz_half.size()));
    } else {
      pop_graph.setInitialValue(
          nz_tensor, poplar::ArrayRef<float>(device_nz_values.data(),
                                             device_nz_values.size()));
    }

    // execute matmul
    auto out =
        static_::denseSparseMatMul(pop_graph, dense_lhs, sparse_rhs, pop_prog,
                                   false, false, debugContext(), opts, &cache);

    setOutTensor(0, out);
  }
};

namespace {
OpxCreator<DenseSparseMatMulOpx>
    DenseSparseMatMulOpxCreator({DenseSparseMatMulOp::defaultOperatorId()});
}

} // namespace popx

// -------------- PyBind --------------
// `dense_sparse_matmul_binding` must equal filename
PYBIND11_MODULE(dense_sparse_matmul_impl, m) {
  // Bindings the parameters of the op: constructor + fields.
  py::class_<DenseSparseMatMulParams>(m, "DenseSparseMatMulParams")
      .def(py::init<std::vector<float>, std::vector<int64_t>,
                    std::vector<int64_t>, int>(),
           py::arg("nz"), py::arg("cols"), py::arg("rows"),
           py::arg("output_cols"))
      .def_readwrite("nz", &DenseSparseMatMulParams::nz)
      .def_readwrite("cols", &DenseSparseMatMulParams::cols)
      .def_readwrite("rows", &DenseSparseMatMulParams::rows)
      .def_readwrite("output_cols", &DenseSparseMatMulParams::output_cols);

  auto cls = popart::ir::op::makeParameterizedOpBindings<DenseSparseMatMulOp>(
      m, "DenseSparseMatMulOp");
  cls.def("setAvailableMemoryProportion",
          &DenseSparseMatMulOp::setAvailableMemoryProportion);
}
// pybind end

} // namespace popart

// -------------- cppimport --------------
// cppimport configuration for compiling the pybind11 module.
// clang-format off
/*
<%
cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall', '-Wno-sign-compare', '-Wno-attributes', '-Wno-narrowing']
cfg['libraries'] = ['popsparse', 'popart']
setup_pybind11(cfg)
%>
*/
