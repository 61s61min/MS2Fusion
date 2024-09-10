# coding:utf-8
import math
from functools import partial
from typing import Callable, Any,Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from matplotlib import pyplot as plt
from timm.layers import DropPath, Mlp
from torch.utils import checkpoint
# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
from torchvision import transforms


class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out
try:
    from models.mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass

# an alternative for mamba_ssm
try:
    from model.selective_scan.selective_scan.selective_scan_interface import selective_scan_fn as selective_scan_fn_v1
except:
    pass

# cross selective scan ===============================
if True:
    import selective_scan_cuda_core as selective_scan_cuda


    class SelectiveScan(torch.autograd.Function):
        # @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        # @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)


    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y

        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None


    class CrossScan_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_rgb: torch.Tensor, x_e: torch.Tensor):
            # B, C, H, W -> B, 2, C, 2 * H * W
            B, C, H, W = x_rgb.shape
            ctx.shape = (B, C, H, W)
            xs_fuse = x_rgb.new_empty((B, 2, C, 2 * H * W))
            xs_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)
            xs_fuse[:, 1] = torch.flip(xs_fuse[:, 0], dims=[-1])
            return xs_fuse

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, 2, d, l)
            B, C, H, W = ctx.shape
            L = 2 * H * W
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # B,  d, 2 * H * W
            # ys = ys[:, 0] + ys[:, 1]  # B, d, 2 * H * W
            # get B, d, H*W
            return ys[:, :, 0:H * W].view(B, -1, H, W), ys[:, :, H * W:2 * H * W].view(B, -1, H, W)


    class CrossMerge_multimodal(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, L = ys.shape
            # ctx.shape = (H, W)
            # ys = ys.view(B, K, D, -1)
            ys = ys[:, 0] + ys[:, 1].flip(dims=[-1])  # B, d, 2 * H * W, broadcast
            # y = ys[:, :, 0:L//2] + ys[:, :, L//2:L]
            return ys[:, :, 0:L // 2], ys[:, :, L // 2:L]

        @staticmethod
        def backward(ctx, x1: torch.Tensor, x2: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            # H, W = ctx.shape
            B, C, L = x1.shape
            xs = x1.new_empty((B, 2, C, 2 * L))
            xs[:, 0] = torch.cat([x1, x2], dim=2)
            xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
            xs = xs.view(B, 2, C, 2 * L)
            return xs, None, None


    def cross_selective_scan(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)

        y = CrossMerge.apply(ys)
        y=y.half()
        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def selective_scan_1d(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        A_logs = A_logs[: A_logs.shape[0] // 4]
        Ds = Ds[: Ds.shape[0] // 4]
        B, D, H, W = x.shape
        D, N = A_logs.shape
        # get 1st of dt_projs_weight
        x_proj_weight = x_proj_weight[0].unsqueeze(0)
        x_proj_bias = x_proj_bias[0].unsqueeze(0) if x_proj_bias is not None else None
        dt_projs_weight = dt_projs_weight[0].unsqueeze(0)
        dt_projs_bias = dt_projs_bias[0].unsqueeze(0) if dt_projs_bias is not None else None
        K, D, R = dt_projs_weight.shape  # K=1
        L = H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # xs = CrossScan.apply(x)
        xs = x.view(B, -1, L).unsqueeze(dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, L)

        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = ys[:, 0].transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


    def cross_selective_scan_multimodal_k1(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        # x_fuse = CrossScan_multimodal.apply(x_rgb, x_e) # B, C, H, W -> B, 1, C, 2 * H * W
        B, C, H, W = x_rgb.shape
        x_fuse = x_rgb.new_empty((B, 1, C, 2 * H * W))
        x_fuse[:, 0] = torch.concat([x_rgb.flatten(2, 3), x_e.flatten(2, 3)], dim=2)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2 * H * W)

        # y = CrossMerge_multimodal.apply(ys)
        y = ys[:, 0, :, 0:L // 2] + ys[:, 0, :, L // 2:L]

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x_rgb.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x_rgb.dtype)

        return y


    def cross_selective_scan_multimodal_k2(
            x_rgb: torch.Tensor = None,
            x_e: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm1: torch.nn.Module = None,
            out_norm2: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x_rgb.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = 2 * H * W

        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        x_fuse = CrossScan_multimodal.apply(x_rgb, x_e)  # B, C, H, W -> B, 2, C, 2 * H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_fuse, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        x_fuse = x_fuse.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            x_fuse, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, 2 * H * W).to(x_rgb.dtype)

        y_rgb, y_e = CrossMerge_multimodal.apply(ys)

        y_rgb = y_rgb.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_e = y_e.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y_rgb = out_norm1(y_rgb).to(x_rgb.dtype)
        y_e = out_norm2(y_e).to(x_e.dtype)

        return y_rgb, y_e


# fvcore flops =======================================

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def print_jit_input_names(inputs):
    # tensor.11, dt.1, A.1, B.1, C.1, D.1, z.1, None
    try:
        print("input params: ", end=" ", flush=True)
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)

    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs")  # (B, D, L)
    assert inputs[1].debugName().startswith("dts")  # (B, D, L)
    assert inputs[2].debugName().startswith("As")  # (D, N)
    assert inputs[3].debugName().startswith("Bs")  # (D, N)
    assert inputs[4].debugName().startswith("Cs")  # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = len(inputs) > 5 and inputs[5].debugName().startswith("z")
    else:
        with_z = len(inputs) > 6 and inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    # flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops
DEV = False

class CM_Attention(nn.Module):
    '''
    Multimodal Mamba Selective Scan 2D
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.conv2d_modalx = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        # x proj; dt proj ============================
        self.K = 2
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.K2 = self.K
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        if not self.softmax_version:
            self.out_norm1 = nn.LayerNorm(self.d_inner)
            self.out_norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner * 2, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.d_inner, self.d_inner // 16, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.d_inner // 16, self.d_inner, bias=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2_multimodal(self, x_rgb: torch.Tensor, x_e: torch.Tensor, nrows=-1):
        return cross_selective_scan_multimodal_k2(
            x_rgb, x_e, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm1", None), getattr(self, "out_norm2", None),
            self.softmax_version,
            nrows=nrows,
        )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb= x_rgb.permute(0,2,3,1).contiguous()
        x_e =x_e.permute(0,2,3,1).contiguous()
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        if self.d_conv > 1:
            x_rgb_trans = x_rgb.permute(0, 3, 1, 2).contiguous()
            x_e_trans = x_e.permute(0, 3, 1, 2).contiguous()
            x_rgb_conv = self.act(self.conv2d(x_rgb_trans))  # (b, d, h, w)
            x_e_conv = self.act(self.conv2d_modalx(x_e_trans))  # (b, d, h, w)
            y_rgb, y_e = self.forward_corev2_multimodal(x_rgb_conv, x_e_conv)  # b, d, h, w -> b, h, w, d
            # SE to get attention, scale
            b, d, h, w = x_rgb_trans.shape
            x_rgb_squeeze = self.avg_pool(x_rgb_trans).view(b, d)
            x_e_squeeze = self.avg_pool(x_e_trans).view(b, d)
            x_rgb_exitation = self.fc1(x_rgb_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()  # b, 1, 1, d
            x_e_exitation = self.fc2(x_e_squeeze).view(b, d, 1, 1).permute(0, 2, 3, 1).contiguous()
            y_rgb = y_rgb * x_e_exitation
            y_e = y_e * x_rgb_exitation
            y = torch.concat([y_rgb, y_e], dim=-1)
        out = self.dropout(self.out_proj(y)).permute(0,3,1,2).contiguous()

        return out



class CMSSF(nn.Module):
    '''
    Concat Mamba (ConMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.op = CM_Attention(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, x: torch.Tensor):
        x_rgb=x[0]
        x_e = x[1]
        x = x_rgb + x_e + self.drop_path(self.op(x_rgb, x_e))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, x: torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)







class DBISSF_Attention(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            Cross=True,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.Cross = Cross
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_share = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        self.dt_proj_share = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.A_log_share = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)
        self.D_share = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)
        self.out_norm_share = nn.LayerNorm(self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def visualization(feat_rgb_A,feat_rgb_B,feat_rgb_C,feat_rgb_D,feat_e_A,feat_e_B,feat_e_C,feat_e_D,feat_fusion,h,w):

        save_dir = 'visualization'
        feat_rgb_A = torch.mean(feat_rgb_A, dim=1)
        feat_rgb_B = torch.mean(feat_rgb_B, dim=1)
        feat_rgb_C = torch.mean(feat_rgb_C, dim=1)
        feat_rgb_D = torch.mean(feat_rgb_D, dim=1)
        feat_e_A = torch.mean(feat_e_A, dim=1)
        feat_e_B = torch.mean(feat_e_B, dim=1)
        feat_e_C = torch.mean(feat_e_C, dim=1)
        feat_e_D = torch.mean(feat_e_D, dim=1)
        block = [feat_rgb_A,feat_rgb_B,feat_rgb_C,feat_rgb_D,feat_e_A,feat_e_B,feat_e_C,feat_e_D]
        black_name = ["feat_rgb_A","feat_rgb_B","feat_rgb_C","feat_rgb_D","feat_e_A","feat_e_B","feat_e_C","feat_e_D"]
        plt.figure()
        for i in range(len(block)):
            feature = transforms.ToPILImage()(block[i].squeeze())
            ax = plt.subplot(3, 3, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(black_name[i], fontsize=8)
            plt.imshow(feature)
        plt.savefig(save_dir + 'fea_{}x{}.png'.format(h, w), dpi=300)

    def forward(self, x_rgb: torch.Tensor,x_share_rgb, x_e: torch.Tensor,x_share_e):
        selective_scan = selective_scan_fn_v1
        B, L, d = x_rgb.shape
        x_rgb = x_rgb.permute(0, 2, 1)
        x_e = x_e.permute(0, 2, 1)
        x_share_rgb = x_share_rgb.permute(0, 2, 1)
        x_share_e = x_share_e.permute(0, 2, 1)

        x_dbl_rgb = self.x_proj_1(rearrange(x_rgb, "b d l -> (b l) d"))  # (bl d)
        x_dbl_e = self.x_proj_2(rearrange(x_e, "b d l -> (b l) d"))  # (bl d)
        x_dbl_share = self.x_proj_share(rearrange(torch.add(x_share_rgb,x_share_e), "b d l -> (b l) d"))  # (bl d)

        dt_rgb, B_rgb, C_rgb = torch.split(x_dbl_rgb, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_share, B_share, C_share = torch.split(x_dbl_share, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt_rgb = self.dt_proj_1.weight @ dt_rgb.t()
        dt_e = self.dt_proj_2.weight @ dt_e.t()
        dt_share = self.dt_proj_share.weight @ dt_share.t()

        dt_rgb = rearrange(dt_rgb, "d (b l) -> b d l", l=L)
        dt_e = rearrange(dt_e, "d (b l) -> b d l", l=L)
        dt_share = rearrange(dt_share, "d (b l) -> b d l", l=L)

        A_rgb = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_e = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        A_share = -torch.exp(self.A_log_share.float())  # (k * d, d_state)

        B_rgb = rearrange(B_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_share = rearrange(B_share, "(b l) dstate -> b dstate l", l=L).contiguous()

        C_rgb = rearrange(C_rgb, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_share = rearrange(C_share, "(b l) dstate -> b dstate l", l=L).contiguous()

        if self.Cross:
            y_rgb = selective_scan(
                x_rgb, dt_rgb,
                A_rgb, B_rgb, C_e, self.D_1.float(),
                delta_bias=self.dt_proj_1.bias.float(),
                delta_softplus=True,
            )
            y_e = selective_scan(
                x_e, dt_e,
                A_e, B_e, C_rgb, self.D_2.float(),
                delta_bias=self.dt_proj_2.bias.float(),
                delta_softplus=True,
            )
            y_share_rgb = selective_scan(
                x_rgb, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus = True,
            )
            y_share_e = selective_scan(
                x_e, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
        else:
            y_rgb = selective_scan(
                x_rgb, dt_rgb,
                A_rgb, B_rgb, C_rgb, self.D_1.float(),
                delta_bias=self.dt_proj_1.bias.float(),
                delta_softplus=True,
            )
            y_e = selective_scan(
                x_e, dt_e,
                A_e, B_e, C_e, self.D_2.float(),
                delta_bias=self.dt_proj_2.bias.float(),
                delta_softplus=True,
            )
            y_share_rgb = selective_scan(
                x_rgb, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
            y_share_e = selective_scan(
                x_e, dt_share,
                A_share, B_share, C_share, self.D_share.float(),
                delta_bias=self.dt_proj_share.bias.float(),
                delta_softplus=True,
            )
        # assert out_y.dtype == torch.float
        y_rgb = rearrange(y_rgb, "b d l -> b l d")
        y_rgb = self.out_norm_1(y_rgb)
        y_e = rearrange(y_e, "b d l -> b l d")
        y_e = self.out_norm_2(y_e)
        y_share_rgb = rearrange(y_share_rgb, "b d l -> b l d")
        y_share_rgb = self.out_norm_share(y_share_rgb)
        y_share_e = rearrange(y_share_e, "b d l -> b l d")
        y_share_e = self.out_norm_share(y_share_e)
        return y_rgb,y_share_rgb, y_e,y_share_e
class DBISSF_SS(nn.Module):
    '''
    Cross Mamba Attention Fusion Selective Scan 2D Module with SSM
    '''

    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            Cross=True,
            learnableweight=False,
            scan = False,
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.Cross = Cross
        self.learnableweight=learnableweight
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_modalx = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj_share = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        if learnableweight:
            self.learnableweight_rgb = LearnableWeights()
            self.learnableweight_e = LearnableWeights()
        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_e = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_share = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_e = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.dropout_share = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.CMA_ssm = DBISSF_Attention(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            Cross=Cross,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,

            **kwargs,
        )
        self.scan = scan
        if scan:
            self.CMA_ssm2 = DBISSF_Attention(
                d_model=self.d_model,
                d_state=self.d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=dt_rank,
                Cross=Cross,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,

                **kwargs,
            )

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor):
        x_rgb = x_rgb.permute(0, 2,3,1).contiguous()
        x_e = x_e.permute(0, 2,3,1).contiguous()
        temp_rgb = x_rgb
        temp_e =x_e
        x_rgb = self.in_proj(x_rgb)
        x_e = self.in_proj_modalx(x_e)
        x_share_rgb =self.in_proj_share(temp_rgb)
        x_share_e = self.in_proj_share(temp_e)

        B, H, W, D = x_rgb.shape
        if self.d_conv > 1:
            x_rgb = x_rgb.permute(0, 3,1,2).contiguous()
            x_e = x_e.permute(0, 3,1,2).contiguous()
            x_share_rgb = x_share_rgb.permute(0, 3, 1, 2).contiguous()
            x_share_e = x_share_e.permute(0, 3, 1, 2).contiguous()

            x_rgb_conv = self.act(self.conv2d(x_rgb))  # (b, d, h, w)
            x_e_conv = self.act(self.conv2d(x_e))  # (b, d, h, w)
            x_share_rgb_conv = self.act(self.conv2d(x_share_rgb))  # (b, d, h, w)
            x_share_e_conv = self.act(self.conv2d(x_share_e))  # (b, d, h, w)

            x_rgb_conv_new = rearrange(x_rgb_conv, "b d h w -> b (h w) d")
            x_e_conv_new = rearrange(x_e_conv, "b d h w -> b (h w) d")
            x_share_rgb_conv_new = rearrange(x_share_rgb_conv, "b d h w -> b (h w) d")
            x_share_e_conv_new = rearrange(x_share_e_conv, "b d h w -> b (h w) d")

            y_rgb, y_share_rgb, y_e, y_share_e = self.CMA_ssm(x_rgb_conv_new,x_share_rgb_conv_new, x_e_conv_new, x_share_e_conv_new)
            # to b, d, h, w
            y_rgb = y_rgb.view(B, H, W, -1)
            y_e = y_e.view(B, H, W, -1)
            y_share_rgb = y_share_rgb.view(B, H, W, -1)
            y_share_e = y_share_e.view(B, H, W, -1)
            if self.scan:
                x_rgb_conv2=x_rgb_conv
                x_e_conv2=x_e_conv
                x_share_rgb_conv2=x_share_rgb
                x_share_e_conv2=x_share_e
                x_rgb_conv_new2 = rearrange(x_rgb_conv2, "b d h w -> b (w h) d")
                x_e_conv_new2 = rearrange(x_e_conv2, "b d h w -> b (w h) d")
                x_share_rgb_conv_new2 = rearrange(x_share_rgb_conv2, "b d h w -> b (w h) d")
                x_share_e_conv_new2 = rearrange(x_share_e_conv2, "b d h w -> b (w h) d")
                y_rgb2, y_share_rgb2, y_e2, y_share_e2 = self.CMA_ssm2(x_rgb_conv_new2, x_share_rgb_conv_new2, x_e_conv_new2,
                                                                  x_share_e_conv_new2)
                y_rgb2 = y_rgb2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_e2 = y_e2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_share_rgb2 = y_share_rgb2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_share_e2 = y_share_e2.view(B, W, H, -1).permute(0,2,1,3).contiguous()
                y_rgb=0.5*y_rgb+0.5*y_rgb2
                y_e = 0.5*y_e+0.5*y_e2
                y_share_rgb=0.5*y_share_rgb+0.5*y_share_rgb2
                y_share_e=0.5*y_share_e+0.5*y_share_e2
        out_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb)).permute(0, 3,1,2).contiguous()
        out_e = self.dropout_e(self.out_proj_e(y_e)).permute(0, 3,1,2).contiguous()
        out_share_rgb = self.dropout_share(self.out_proj_share(y_share_rgb)).permute(0, 3, 1, 2).contiguous()
        out_share_e = self.dropout_share(self.out_proj_share(y_share_e)).permute(0, 3, 1, 2).contiguous()

        return out_rgb, out_share_rgb, out_e, out_share_e


class DBISSF(nn.Module):
    '''
    Cross Mamba Fusion (CroMB) fusion, with 2d SSM
    '''

    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            return_separately:bool = True,
            Cross:bool = True,
            scan:bool = False,
            learnableweight:bool=False,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        # self.norm = norm_layer(hidden_dim)
        self.return_serparately = return_separately
        self.op = DBISSF_SS(
            d_model=hidden_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            shared_ssm=shared_ssm,
            softmax_version=softmax_version,
            Cross = Cross,
            learnableweight=learnableweight,
            scan=scan,
            **kwargs
        )
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)
        self.drop_path_share = DropPath(drop_path)
        # LearnableCoefficient

        self.return_serparately=return_separately
        if not return_separately:
            self.learnable_weight = LearnableWeights()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                           channels_first=False)

    def _forward(self, x: torch.Tensor):
        x_rgb=x[0]
        x_e =x[1]
        x_rgb_cross,x_share_rgb, x_e_cross,x_share_e = self.op(x_rgb, x_e)

        b,d,h,w= x_e.shape
        x_rgb_new = x_rgb + self.drop_path1(x_rgb_cross)
        x_e_new = x_e + self.drop_path2(x_e_cross)
        x_share_rgb = x_share_rgb +self.drop_path_share(x_share_rgb)
        x_share_e = x_share_e + self.drop_path_share(x_share_e)
        if self.return_serparately:
            return x_rgb_new,x_share_rgb, x_e_new, x_share_e
        else:
            x_cross_feat = self.learnable_weight(x_rgb_new,x_e_new)
        #=================visualization==================
        # save_dir = 'visualization'
        # fea_rgb = torch.mean(x_rgb, dim=1)
        # fea_rgb_CFE = torch.mean(x_rgb_cross, dim=1)
        # fea_rgb_new = torch.mean(x_rgb_new, dim=1)
        # fea_ir = torch.mean(x_e, dim=1)
        # fea_ir_CFE = torch.mean(x_e_cross, dim=1)
        # fea_ir_new = torch.mean(x_e_new, dim=1)
        # fea_new = torch.mean(x_cross_feat, dim=1)
        # block = [fea_rgb, fea_rgb_CFE, fea_rgb_new, fea_ir, fea_ir_CFE, fea_ir_new, fea_new]
        # black_name = ['fea_rgb', 'fea_rgb After cross', 'fea_rgb new', 'fea_ir', 'fea_ir After cross', 'fea_ir new',
        #               'fea_ir mambafusion']
        # plt.figure()
        # for i in range(len(block)):
        #     feature = transforms.ToPILImage()(block[i].squeeze())
        #     ax = plt.subplot(3, 3, i + 1)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #     ax.set_title(black_name[i], fontsize=8)
        #     plt.imshow(feature)
        #     plt.show()
        # plt.savefig(save_dir + '/fea_{}x{}.png'.format(h, w), dpi=300)
            return x_cross_feat

    def forward(self, x:torch.Tensor):
        '''
        B C H W, B C H W -> B C H W
        '''

        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
class SSF(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            Cross:bool = True,
            scan: bool = False,
            use_learnableweights: bool = False,

            separate_return: bool = True,
            only_cm:bool = False,
            only_dbi:bool=False,

            # =================================================
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank: Any = "auto",
            ssm_ratio=2.0,
            shared_ssm=False,
            softmax_version=False,
            use_checkpoint: bool = False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            **kwargs,
    ):
        super().__init__()
        self.only_cm = only_cm
        self.only_dbi = only_dbi
        if only_cm:
            self.cm_fusion = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
            )
        if only_dbi:
            self.dbi_fusion=DBISSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
                return_separately=separate_return,
                learnableweight=use_learnableweights,
                scan=scan,
                Cross=Cross
            )
        if not only_dbi and not only_cm:
            self.cm_fusion = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.cm_fusion_rgb = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.cm_fusion_e = CMSSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4
            )
            self.dbi_fusion = DBISSF(
                hidden_dim=hidden_dim,
                mlp_ratio=0.0,
                d_state=4,
                return_separately=separate_return,
                learnableweight=use_learnableweights,
                scan=scan,
                Cross=Cross
            )

    def forward(self,x):
        # VSSBlock=======open=====
        # temp =[]
        # temp.append(x[0].permute(0,3,1,2).contiguous())
        # temp.append(x[1].permute(0, 3,1,2).contiguous())
        # x=temp
        # ==========
        if self.only_dbi:
            x_feat=self.dbi_fusion(x)
        elif self.only_cm:
            x_feat = self.cm_fusion(x)
        elif not self.only_cm and not self.only_dbi:
            x_cross_rgb, x_share_rgb, x_cross_e, x_share_e = self.dbi_fusion(x)
            x_fuse_rgb = []
            x_fuse_e = []

            x_fuse_rgb.append(x_cross_rgb)
            x_fuse_rgb.append(x_share_rgb)

            x_fuse_e.append(x_cross_e)
            x_fuse_e.append(x_share_e)

            x_rgb = self.cm_fusion_rgb(x_fuse_rgb)
            x_e = self.cm_fusion_e(x_fuse_e)

            x_new=[]
            x_new.append(x_rgb)
            x_new.append(x_e)
            # x_new=[]
            # x_new.append(x_cross_rgb)
            # x_new.append(x_cross_e)
            x_feat = self.cm_fusion(x_new)
        return x_feat

class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.2)


class MiniInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiniInception, self).__init__()
        self.conv1_left  = ConvBnLeakyRelu2d(in_channels,   out_channels//2)
        self.conv1_right = ConvBnLeakyRelu2d(in_channels,   out_channels//2, padding=2, dilation=2)
        self.conv2_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv2_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
        self.conv3_left  = ConvBnLeakyRelu2d(out_channels,  out_channels//2)
        self.conv3_right = ConvBnLeakyRelu2d(out_channels,  out_channels//2, padding=2, dilation=2)
    def forward(self,x):
        x = torch.cat((self.conv1_left(x), self.conv1_right(x)), dim=1)
        x = torch.cat((self.conv2_left(x), self.conv2_right(x)), dim=1)
        x = torch.cat((self.conv3_left(x), self.conv3_right(x)), dim=1)
        return x


class MFNet(nn.Module):

    def __init__(self, n_class):
        super(MFNet, self).__init__()
        rgb_ch = [16,48,48,96,96]
        inf_ch = [16,48,48,96,96]
        # inf_ch =[16,16,16,36,36]
        self.conv1_rgb   = ConvBnLeakyRelu2d(3, rgb_ch[0])
        self.conv2_1_rgb = ConvBnLeakyRelu2d(rgb_ch[0], rgb_ch[1])
        self.conv2_2_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[1])
        self.conv3_1_rgb = ConvBnLeakyRelu2d(rgb_ch[1], rgb_ch[2])
        self.conv3_2_rgb = ConvBnLeakyRelu2d(rgb_ch[2], rgb_ch[2])
        self.conv4_rgb   = MiniInception(rgb_ch[2], rgb_ch[3])
        self.conv5_rgb   = MiniInception(rgb_ch[3], rgb_ch[4])

        self.conv1_inf   = ConvBnLeakyRelu2d(1, inf_ch[0])
        self.conv2_1_inf = ConvBnLeakyRelu2d(inf_ch[0], inf_ch[1])
        self.conv2_2_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[1])
        self.conv3_1_inf = ConvBnLeakyRelu2d(inf_ch[1], inf_ch[2])
        self.conv3_2_inf = ConvBnLeakyRelu2d(inf_ch[2], inf_ch[2])
        self.conv4_inf   = MiniInception(inf_ch[2], inf_ch[3])
        self.conv5_inf   = MiniInception(inf_ch[3], inf_ch[4])

        self.decode4     = ConvBnLeakyRelu2d(rgb_ch[3]+inf_ch[3], rgb_ch[2]+inf_ch[2])
        self.decode3     = ConvBnLeakyRelu2d(rgb_ch[2]+inf_ch[2], rgb_ch[1]+inf_ch[1])
        self.decode2     = ConvBnLeakyRelu2d(rgb_ch[1]+inf_ch[1], rgb_ch[0]+inf_ch[0])
        self.decode1     = ConvBnLeakyRelu2d(rgb_ch[0]+inf_ch[0], n_class)
        self.fuse = SSF(rgb_ch[3],True,False,False,True,False,False)
        self.conv1x1= nn.Conv2d(in_channels=rgb_ch[3],out_channels=rgb_ch[3]+inf_ch[3],kernel_size=1)

    def forward(self, x):
        # split data into RGB and INF
        x_rgb = x[:,:3]
        x_inf = x[:,3:]

        # encode
        x_rgb    = self.conv1_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb, kernel_size=2, stride=2) # pool1
        x_rgb    = self.conv2_1_rgb(x_rgb)
        x_rgb_p2 = self.conv2_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p2, kernel_size=2, stride=2) # pool2
        x_rgb    = self.conv3_1_rgb(x_rgb)
        x_rgb_p3 = self.conv3_2_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p3, kernel_size=2, stride=2) # pool3
        x_rgb_p4 = self.conv4_rgb(x_rgb)
        x_rgb    = F.max_pool2d(x_rgb_p4, kernel_size=2, stride=2) # pool4
        x_rgb    = self.conv5_rgb(x_rgb)

        x_inf    = self.conv1_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf, kernel_size=2, stride=2) # pool1
        x_inf    = self.conv2_1_inf(x_inf)
        x_inf_p2 = self.conv2_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p2, kernel_size=2, stride=2) # pool2
        x_inf    = self.conv3_1_inf(x_inf)
        x_inf_p3 = self.conv3_2_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p3, kernel_size=2, stride=2) # pool3
        x_inf_p4 = self.conv4_inf(x_inf)
        x_inf    = F.max_pool2d(x_inf_p4, kernel_size=2, stride=2) # pool4
        x_inf    = self.conv5_inf(x_inf)
        x=[]
        x.append(x_rgb)
        x.append(x_inf)
        x=self.fuse(x)
        x=self.conv1x1(x)
        # x = torch.cat((x_rgb, x_inf), dim=1) # fusion RGB and INF

        # decode

        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool4
        x = self.decode4(x + torch.cat((x_rgb_p4, x_inf_p4), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool3
        x = self.decode3(x + torch.cat((x_rgb_p3, x_inf_p3), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool2
        x = self.decode2(x + torch.cat((x_rgb_p2, x_inf_p2), dim=1))
        x = F.upsample(x, scale_factor=2, mode='nearest') # unpool1
        x = self.decode1(x)

        return x


def unit_test():
    import numpy as np
    device="cuda"
    x = torch.tensor(np.random.rand(2,4,640,640).astype(np.float32)).to(device)
    model = MFNet(n_class=9).to(device)
    y = model(x)
    print('output shape:', y.shape)
    assert y.shape == (2,9,640,640), 'output shape (2,9,640,640) is expected!'
    print('test ok!')


if __name__ == '__main__':
    unit_test()
