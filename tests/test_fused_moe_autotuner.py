"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
from torch.nn import functional as F

import flashinfer
import flashinfer.fused_moe as fused_moe
from flashinfer import fp4_quantize
from flashinfer.autotuner import AutoTuner, autotune
import logging
logger = logging.getLogger(__name__)
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_DTYPE = torch.float8_e4m3fn
REF_CHECK = False 

def dynamic_per_tensor_fp8_quant(x: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
    fp8_traits_max = FLOAT8_E4M3_MAX
    fp8_traits_min = -FLOAT8_E4M3_MAX
    fp8_max = torch.tensor(fp8_traits_max).float()
    one = torch.tensor(1.0).float()

    x_max = x.abs().max().float()
    scale = x_max / fp8_max
    iscale = one / scale
    out = (x.float() * iscale).clamp(fp8_traits_min, fp8_traits_max).to(FP8_DTYPE)
    return out, scale.view((1,))


def gen_tensor(shape, dtype, stype=None, scale=1.0):
    x = torch.randn(*shape, dtype=dtype).cuda() * scale
    return x.to(stype) if stype else x


def cast_to_representable(x):
    x_q, x_scale = dynamic_per_tensor_fp8_quant(x)
    x = x_q.to(x.dtype) * x_scale.to(x.dtype)
    return x


def convert_swizzled_to_linear(a_sf_swizzled: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_swizzled, (1, m_tiles, k_tiles, 32, 4, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]


def dequantize_nvfp4_to_dtype(
    tensor_fp4, tensor_sf, global_scale, dtype, device, block_size=16
):
    """Dequantize the fp4 tensor back to high precision."""
    # Two fp4 values are packed into one uint8.
    assert tensor_fp4.dtype == torch.uint8
    m, packed_k = tensor_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(tensor_fp4, dtype)
    tensor_f32 = tensor_f32.reshape(m, k // block_size, block_size)
    tensor_sf = tensor_sf.view(torch.float8_e4m3fn)
    tensor_sf = convert_swizzled_to_linear(tensor_sf, m, k, block_size)
    tensor_sf_dtype = tensor_sf.to(torch.float32) / global_scale

    # scale the tensor
    out = (tensor_f32 * tensor_sf_dtype.unsqueeze(-1)).reshape(m, k)
    return out.to(dtype=dtype)


def break_fp4_bytes(a, dtype):
    assert a.dtype == torch.uint8
    m, n = a.shape

    # Vectorized nibble processing
    a_flat = a.flatten()
    high = (a_flat & 0xF0) >> 4  # Upper nibbles
    low = a_flat & 0x0F  # Lower nibbles

    # Combine nibbles for batch processing
    combined = torch.stack((low, high), dim=1).flatten()

    # Vectorized sign and magnitude extraction
    signs = (combined & 0x08).to(torch.bool)  # Sign bits
    abs_vals = (combined & 0x07).to(torch.long)  # Magnitude indices

    # Device-aware lookup and sign application
    kE2M1ToFloat = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    kE2M1 = kE2M1ToFloat.to(device=a.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)

    # Reshape to final form
    return values.reshape(m, n * 2).to(dtype=dtype)


def compute_routing(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute routing weights and selected experts from router logits.

    Args:
        router_logits (torch.Tensor): Router logits of shape [batch_size, num_experts]
        top_k (int): Number of experts to route to per token

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routing_weights: Expert weights of shape [batch_size, top_k]
            - selected_experts: Expert indices of shape [batch_size, top_k]
    """
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.float()
    return routing_weights, selected_experts


def torch_moe_nvfp4(a, w1, w2, topk, topk_weight, topk_ids):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    # score = torch.softmax(score, dim=-1, dtype=torch.float32)
    # topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    # w1 needs to be swapped in terms of gate and up_proj

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            m = w1[i].shape[0]
            assert m % 2 == 0
            w1_expert, w3_expert = w1[i][m // 2 :, :], w1[i][: m // 2, :]
            inter = F.silu(a[mask] @ w1_expert.t()) * (a[mask] @ w3_expert.t())
            inter_gs = torch.tensor(1.0).cuda()
            inter_q, inter_blockscale = fp4_quantize(inter, inter_gs)
            inter = dequantize_nvfp4_to_dtype(
                inter_q,
                inter_blockscale,
                inter_gs,
                dtype=inter.dtype,
                device=inter.device,
                block_size=16,
            ).cuda()
            out[mask] = inter @ w2[i].transpose(0, 1)
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def compute_with_experts(
    num_experts, x, w31_weight, w2_weight, selected_experts, routing_weights
):
    results = torch.zeros_like(x)
    for expert_id in range(num_experts):
        mask = selected_experts == expert_id
        if not mask.sum():
            continue
        batch_idx, nth_expert = torch.where(mask)
        w31_expert = w31_weight[expert_id]  # [2 * intermediate_size, hidden_size]
        w2_expert = w2_weight[expert_id]  # [hidden_size, intermediate_size]

        # Split w13 into w1 and w3
        w3_expert, w1_expert = torch.chunk(w31_expert, 2, dim=0)

        expert_inputs = x[batch_idx]
        inter = F.silu(expert_inputs @ w1_expert.t()) * (expert_inputs @ w3_expert.t())
        output = inter @ w2_expert.t()
        results[batch_idx] += routing_weights[batch_idx, nth_expert, None] * output
    return results.view_as(x)


# Test configurations
BATCH_SIZES = [
    1,4,6,8,12,16,24,32,48,64,96,128,1,4,6,8,12,16,24,32,48,64,96,128,
]
HIDDEN_SIZES = [
    7168,
]
NUM_EXPERTS = [2]
TOP_K_VALUES = [2]
INTERMEDIATE_SIZES = [
   256 ,
]
EP_NUM_EXPERTS = [8]
EP_TOP_K = [2]


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_K_VALUES)
@pytest.mark.parametrize("intermediate_size", INTERMEDIATE_SIZES)
@pytest.mark.parametrize(
    "otype, wtype",
    [(torch.float16, torch.float8_e4m3fn), (torch.bfloat16, torch.float8_e4m3fn)],
)
@pytest.mark.parametrize("quantized_input", [False, True])
def test_moe_nvfp4(
    batch_size,
    hidden_size,
    num_experts,
    top_k,
    intermediate_size,
    otype,
    wtype,
    quantized_input,
    is_autotune,
):
    print(f"batch_size: {batch_size}, hidden_size: {hidden_size}, num_experts: {num_experts}, top_k: {top_k}, intermediate_size: {intermediate_size}, otype: {otype}, wtype: {wtype}, quantized_input: {quantized_input}")
    # Skip invalid configurations
    if top_k > num_experts:
        pytest.skip(
            f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
        )

    torch.manual_seed(42)
    quant_blocksize = 16
    round_up = lambda x, y: (x + y - 1) // y * y
    e = num_experts
    m = batch_size
    n = intermediate_size
    k = hidden_size

    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=otype) / 10
    w1_cutlass = torch.cat((w1[:, n:, :], w1[:, :n, :]), dim=1).contiguous()

    sf_w1_2n = round_up(2 * n, 128)
    sf_w1_k = round_up(k // quant_blocksize, 4)
    w1_blockscale = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_blockscale_cutlass = torch.empty(
        (e, sf_w1_2n, sf_w1_k), device="cuda", dtype=torch.float8_e4m3fn
    )

    w2 = torch.randn((e, k, n), device="cuda", dtype=otype) / 10
    sf_w2_k = round_up(k, 128)
    sf_w2_n = round_up(n // quant_blocksize, 4)
    w2_blockscale = torch.empty(
        (e, sf_w2_k, sf_w2_n), device="cuda", dtype=torch.float8_e4m3fn
    )
    w1_q = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w1_q_cutlass = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
    w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
    w1_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
    w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

    for expert in range(e):
        w1_amax = torch.abs(w1).max().to(torch.float32)
        w2_amax = torch.abs(w2).max().to(torch.float32)
        w1_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[expert] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax

        w1_q[expert], w1_blockscale[expert] = fp4_quantize(w1[expert], w1_gs[expert])

        w1_q_cutlass[expert], w1_blockscale_cutlass[expert] = fp4_quantize(
            w1_cutlass[expert], w1_gs[expert]
        )

        w2_q[expert], w2_blockscale[expert] = fp4_quantize(w2[expert], w2_gs[expert])

    x = torch.randn(m, k, dtype=otype).cuda()
    a1_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.abs(x).max().to(
        torch.float32
    ).cuda()
    a1_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    a2_gs = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    router_logits = torch.randn(m, e, dtype=otype).cuda()
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    # quant_scales format
    # auto const fc1_act_global = quant_scales.value()[0];
    # auto const fc1_weight_block = quant_scales.value()[1];
    # auto const fc1_global = quant_scales.value()[2];
    # auto const fc2_act_global = quant_scales.value()[3];
    # auto const fc2_weight_block = quant_scales.value()[4];
    # auto const fc2_global = quant_scales.value()[5];
    flash_output = torch.zeros_like(x)

    quant_scales = [
        a1_gs,
        w1_blockscale.view(torch.int32),
        1.0 / (a1_gs * w1_gs),
        a2_gs,
        w2_blockscale.view(torch.int32),
        1.0 / (a2_gs * w2_gs),
    ]
    hidden_states = x
    input_sf = None
    if quantized_input:
        hidden_states, input_sf = fp4_quantize(x, a1_gs
        )
    with autotune(is_autotune):
        print(f"is_autotune: {is_autotune}. starting the kernel run")
        import torch.cuda.nvtx as nvtx
        
        batchsize = hidden_states.shape[0]
        with nvtx.range(f"bs_{batchsize}_is_autotune_{is_autotune}"):
            _ = fused_moe.cutlass_fused_moe(
                hidden_states,
                selected_experts.to(torch.int),
                routing_weights,
                w1_q.contiguous().view(torch.long),
                w2_q.contiguous().view(torch.long),
                otype,
                quant_scales=quant_scales,
                input_sf=input_sf,
                output=flash_output,
            )
        print(f"finished the kernel run")
        # Ref check
    if REF_CHECK:
        a_fp4, a_scale_interleaved = fp4_quantize(x, a1_gs)
        _, m_k = a_fp4.shape
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a1_gs,
            dtype=otype,
            device=x.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=otype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=otype)

        for idx in range(0, e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx],
                w1_blockscale[idx],
                w1_gs[idx],
                dtype=w1.dtype,
                device=w1.device,
                block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx],
                w2_blockscale[idx],
                w2_gs[idx],
                dtype=w2.dtype,
                device=w2.device,
                block_size=quant_blocksize,
            )

        w1_q_cutlass = torch.cat((w1_q[:, n:, :], w1_q[:, :n, :]), dim=1).contiguous()
        w1_blockscale_cutlass = torch.cat(
            (w1_blockscale[:, n:, :], w1_blockscale[:, :n, :]), dim=1
        ).contiguous()
        ref_output = torch_moe_nvfp4(
            a_in_dtype, w1_d, w2_d, top_k, routing_weights, selected_experts
        )
        torch.testing.assert_close(ref_output, flash_output, rtol=2e-1, atol=2e-1)




if __name__ == "__main__":
    # pytest.main([__file__, "-v"])
    TEST_BS = [1,4,6,8,12,16,24,32,48,64,96,128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]
    # TEST_BS = [1,48,64,96,128]
    AutoTuner.get().clear_cache()
    for bs in TEST_BS:
        test_moe_nvfp4(
            batch_size=bs,
            hidden_size=7168,
            num_experts=256,
            top_k=2,
            intermediate_size=256,
            otype=torch.bfloat16,
            wtype=torch.float8_e4m3fn,
            quantized_input=False,
            is_autotune=True
            )
        test_moe_nvfp4(
            batch_size=bs,
            hidden_size=7168,
            num_experts=256,
            top_k=2,
            intermediate_size=256,
            otype=torch.bfloat16,
            wtype=torch.float8_e4m3fn,
            quantized_input=False,
            is_autotune=False
            )
        test_moe_nvfp4(
            batch_size=bs,
            hidden_size=7168,
            num_experts=256,
            top_k=2,
            intermediate_size=256,
            otype=torch.bfloat16,
            wtype=torch.float8_e4m3fn,
            quantized_input=False,
            is_autotune=False
            )
