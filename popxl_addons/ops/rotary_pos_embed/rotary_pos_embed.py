# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

# Auto compile cpp files
import cppimport.import_hook

# You need to use `from . import` here and then in the directory `__init__.py` include the necessary functions
from . import rotary_pos_embed_binding
import numpy as np

from typing import Optional, Tuple, Any

import popxl
from popxl.context import op_debug_context, get_current_context
from popxl.ops.utils import check_in_graph, check_tensor_ipu_and_tile_set

__all__ = ["rotary_pos_embed", "trig_tables", "trig_table_constants"]


def trig_tables(
    seq_len: int, rotary_dim: int, base: int = 10000, np_dtype: Any = "float32"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cosΦ, sinΦ np data of shape (seq_len, rotary_dim), where
    Φ = m * θ_i,
    θ_i = base^(-2i/rotary_dim), i = 0 ... rotary_dim
    """
    theta = base ** (-1 * np.arange(0, rotary_dim, 2).astype(np_dtype) / rotary_dim)  # θ_i
    seq_idx = np.arange(seq_len).astype(np_dtype)  # m (sequence index)
    phi = np.outer(seq_idx, theta)
    cos_np = np.cos(phi)  # [cos(mθ_0), cos(mθ_0), cos(mθ_1), cos(mθ_1), ...] (m is a vector)
    sin_np = np.sin(phi)  # [sin(mθ_0), sin(mθ_0), sin(mθ_1), sin(mθ_1), ...]
    return sin_np, cos_np


def trig_table_constants(seq_len: int, rotary_dim: int, base: int = 10000, dtype: popxl.dtypes.dtype = popxl.float32):
    """
    Generate cosΦ, sinΦ constants of shape (seq_len, rotary_dim), where
    Φ = m * θ_i,
    θ_i = base^(-2i/rotary_dim), i = 0 ... rotary_dim
    """
    sin, cos = trig_tables(seq_len, rotary_dim, base, dtype.as_numpy())
    return popxl.constant(sin, dtype), popxl.constant(cos, dtype)


@op_debug_context
def rotary_pos_embed(
    t: popxl.Tensor, sin: popxl.Tensor, cos: popxl.Tensor, rotary_dim: Optional[int] = None
) -> popxl.Tensor:
    """Rotary positional embeddings (RoPE) as described in "RoFormer: Enhanced Transformer with Rotary
        Position Embedding" https://arxiv.org/pdf/2104.09864.pdf

        RoPE is applied to the query and key activations before calculating the score in the attention layer. i.e.
        Score_with_RoPE = RoPE(xQ) @ RoPE(xK).T

        RoPE rotates each token by an angle dependent its sequence position. The score for each token pair is
        then a function of the relative difference between the two tokens.

        RoPE considers the head hidden space (hh) as hh/2 complex numbers:
            [x0,x1,x2,x3, ....x_hh] = [x0_real, x0_img, x1_real, x1_img, ... ]
        Each element in the complex hh is rotated by an angle Φ where
        Φ = m * θ_i where m is the sequence position and θ is a set of angles for each complex number in hh defined
        below.

        RoPE(xQ) @ RoPE(xK).T is equivalent to the Hermitian inner product between two complex tensors - whereby the
        complex conjure of the second tensor gets taken. This means the resultant score is a function of the difference
        of the two angels and therefore relative sequence position.

        θ_i = base^(-2i/hh) where base is a chosen constant and i is the ith hidden space complex number

    Args:
        t (popxl.Tensor): Tensor to rotate
        sin (popxl.Tensor): sin table generated by trig_tables
        cos (popxl.Tensor): cos table generated by trig_tables
        rotary_dim (Optional[int], optional): Number of hidden elements to rotate. Defaults to all.

    Returns:
        popxl.Tensor: Rotated result
    """
    ctx = get_current_context()
    g = ctx.graph
    pb_g = g._pb_graph

    if not rotary_dim:
        rotary_dim = t.shape[-1]

    assert t.rank == 4
    if sin.rank == 2:
        sin = sin.reshape((1, *sin.shape))
    if cos.rank == 2:
        cos = cos.reshape((1, *cos.shape))
    assert sin.rank == 3
    assert cos.rank == 3

    check_in_graph(g, t=t, sin=sin, cos=cos)
    check_tensor_ipu_and_tile_set(t=t, sin=sin, cos=cos)

    settings = ctx._get_op_settings("RotaryPosEmbedOp")
    op = rotary_pos_embed_binding.RotaryPosEmbedOp.createOpInGraph(
        pb_g,
        {0: t.id, 1: sin.id, 2: cos.id},
        {
            0: g._create_tensor_id("t_rotated"),
        },
        rotary_dim,
        settings,
    )
    ctx._op_created(op)

    return popxl.Tensor._from_pb_tensor(op.outTensor(0))
