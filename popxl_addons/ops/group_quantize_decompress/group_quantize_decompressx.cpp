// Copyright (c) 2023 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>
#include <poplar/Tensor.hpp>

#include <poplar/DebugContext.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/exceptions.hpp>

#include <gcl/Collectives.hpp>
#include <poplar/Program.hpp>
#include <poplar/TensorCloneMethod.hpp>
#include <poplar/Type.hpp>
#include <poplin/MatMul.hpp>
#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Encoding.hpp>
#include <popops/Expr.hpp>
#include <popops/Fill.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/SelectScalarFromRows.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>
#include <poputil/Util.hpp>
#include <poputil/exceptions.hpp>

#include "common.hpp"
#include "group_quantize_decompress.hpp"
#include "group_quantize_decompressx.hpp"

#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace popart {
namespace popx {

namespace {

namespace pe = popops::expr;

poplar::Tensor decompressPacked4BitTensor(poplar::Graph &graph,
                                          poplar::Tensor &x,
                                          poplar::Tensor &groupScale,
                                          poplar::Tensor &groupBias,
                                          poplar::program::Sequence &prog) {

  std::vector<size_t> unp_shape{x.shape()[0], x.shape()[1], x.shape()[2] * 4};
  std::vector<size_t> out_shape{x.shape()[0], x.shape()[1] * x.shape()[2] * 4};

  poplar::DebugContext debugContext;

  auto x_unpacked =
      graph.addVariable(poplar::HALF, unp_shape, {debugContext, "x_unpacked"});
  auto computeSet = graph.addComputeSet({debugContext, "unpack4bit"});
  auto mapping = graph.getTileMapping(x);
  auto numWorkers = 6;

  for (auto tile = 0u; tile < mapping.size(); ++tile) {
    for (auto i : mapping[tile]) {
      // Colocate unpacked tensor to input
      graph.setTileMapping(
          x_unpacked.flatten().slice(i.begin() * 4, i.end() * 4), tile);
      // Get constants for slicing input across 6 threads
      auto interval = (i.end() - i.begin());
      auto numElmsPerWorkerNoRemainder = interval / numWorkers;
      auto numElmsRemainder = interval % numWorkers;
      int slice_bounds[7] = {0};
      slice_bounds[0] = i.begin();

      for (auto wid = 0; wid < numWorkers; ++wid) {
        // Determine slice bounds for thread worker
        slice_bounds[wid + 1] = slice_bounds[wid] + numElmsPerWorkerNoRemainder;
        if (wid < numElmsRemainder) {
          ++slice_bounds[wid + 1];
        }
        // add vertex to thread
        auto vertex = graph.addVertex(
            computeSet, "DecompressPacked4BitTensor",
            {{"input",
              x.flatten().slice(slice_bounds[wid], slice_bounds[wid + 1])},
             {"output",
              x_unpacked.flatten().slice(slice_bounds[wid] * 4,
                                         slice_bounds[wid + 1] * 4)}});
        graph.setTileMapping(vertex, tile);
        graph.setPerfEstimate(
            vertex,
            100 + 30 * i.size()); // guess from godbolt per int32 + overhead
      }
    }
  }
  prog.add(poplar::program::Execute(computeSet));

  // Scale float16 tensor (TODO: optimize with scaledAddTo or fuse into vertex)
  auto x_scaled = popops::map(graph, pe::_1 * pe::_2 + pe::_3,
                              {x_unpacked, groupScale, groupBias}, prog);
  auto x_out = x_scaled.reshape(out_shape);

  return x_out;
}

} // namespace

/////////////////////////////////////////////////////////////
/// Forwards opx

GroupQuantizeDecompressOpx::GroupQuantizeDecompressOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<GroupQuantizeDecompressOp>(op, {GroupQuantizeDecompress});
}

void GroupQuantizeDecompressOpx::grow(poplar::program::Sequence &prog) const {

  auto x = getInTensor(0);
  auto groupMax = getInTensor(1);
  auto groupMin = getInTensor(2);

  std::string file{__FILE__};
  std::string directory{file.substr(0, file.rfind("/"))};

  graph().addCodelets(directory + "/group_quantize_decompress_codelet.cpp");
  poplar::Tensor x_decompressed =
      decompressPacked4BitTensor(graph(), x, groupMax, groupMin, prog);

  setOutTensor(0, x_decompressed);
}

/////////////////////////////////////////////////////////////

namespace {
popx::OpxCreator<GroupQuantizeDecompressOpx>
    GroupQuantizeDecompressOpxCreator(GroupQuantizeDecompress);
} // namespace

} // namespace popx
} // namespace popart