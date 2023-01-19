# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import numpy as np
import pytest
import torch
from transformers.models.gptj.modeling_gptj import fixed_pos_embedding, apply_rotary_pos_emb

import popxl
from popxl import ops
import popxl_addons as addons
from popxl_addons.ops.rotary_pos_embed import trig_tables, rotary_pos_embed
from popxl.utils import to_numpy


def HF_rope(Q, sincos, rotary_dim):
    if rotary_dim is not None:
        q_rot = Q[:, :, :, :rotary_dim]
        q_pass = Q[:, :, :, rotary_dim:]

        q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=0)

        q_rot = torch.cat([q_rot, q_pass], dim=-1)
    else:
        q_rot = apply_rotary_pos_emb(Q, sincos, offset=0)

    return q_rot


@pytest.mark.parametrize("rotary_dim", (0, 4, 16, 32))
def test_rotary_embed(rotary_dim):
    seq = 8
    batch = 3
    heads = 2
    hdim = 32  # hidden size per head

    Q = torch.rand((batch, seq, heads, hdim))
    Q.requires_grad_()

    rotary_dim = rotary_dim or Q.shape[-1]
    sincos = fixed_pos_embedding(Q[:, :, :, :rotary_dim], 1, seq_len=seq)

    # test trig tables
    sin_np, cos_np = trig_tables(seq, rotary_dim)
    np.testing.assert_allclose(sin_np, sincos[0], rtol=1e-6)
    np.testing.assert_allclose(cos_np, sincos[1], rtol=1e-6)

    q_pt = HF_rope(Q, sincos, rotary_dim)

    q_grad = torch.rand((batch, seq, heads, hdim))
    q_pt.backward(q_grad)

    ir = popxl.Ir()
    with ir.main_graph:
        q_xl = popxl.variable(to_numpy(Q.detach()))
        sin, cos = popxl.constant(to_numpy(sincos[0])), popxl.constant(to_numpy(sincos[1]))

        rot_graph = ir.create_graph(rotary_pos_embed, q_xl, sin, cos, rotary_dim)
        drot_info = popxl.transforms.autodiff(rot_graph, grads_required=rot_graph.inputs[:1])
        info = ops.call_with_info(rot_graph, q_xl, sin, cos)
        q_xl = info.outputs[0]

        (dq_xl,) = ops.call(drot_info.graph, popxl.constant(to_numpy(q_grad)), inputs_dict=drot_info.inputs_dict(info))

        q_d2h = addons.host_store(q_xl)
        dq_d2h = addons.host_store(dq_xl)

    with popxl.Session(ir) as sess:
        out = sess.run()

    np.testing.assert_allclose(q_pt.detach(), out[q_d2h])
    np.testing.assert_allclose(Q.grad, out[dq_d2h])
