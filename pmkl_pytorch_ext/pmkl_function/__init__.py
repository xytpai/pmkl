import torch
import torch.nn as nn
from torch.autograd import Function
import pmkl_pytorch_ext


__all__ = [
    'fused_self_attention',
]


class fused_self_attention_(Function):
    @staticmethod
    def forward(q, k, v, is_casual, debug=False):
        if q.is_cpu and debug:
            return pmkl_pytorch_ext.fused_self_attention_fw(q, k, v, is_casual)
        else:
            raise NotImplementedError()
    @staticmethod
    def setup_context(ctx, inputs, output):
        q, k, v, is_casual, debug = inputs
        out, out_m, out_l = output
        ctx.save_for_backward(q, k, v, torch.BoolTensor([is_casual]), torch.BoolTensor([debug]), out_m, out_l)
    @staticmethod
    def backward(ctx, grad_output, grad_m, grad_l):
        q, k, v, is_casual, debug, out_m, out_l = ctx.saved_tensors
        if q.is_cpu and debug.item():
            dq, dk, dv = pmkl_pytorch_ext.fused_self_attention_bw(
                grad_output, q, k, v, out_m, out_l, is_casual.item())
        else:
            raise NotImplementedError()
        return dq, dk, dv, None, None
fused_self_attention = fused_self_attention_.apply
