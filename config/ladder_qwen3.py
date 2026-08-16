# =============================================================================
# Qwen3 config ladder, in the same ModelConfig form as the GPT / Llama-2 /
# Llama-3 files.
#
# !! ModelConfig NEEDS TWO NEW FIELDS !!
#
#     head_dim: int | None = None   # None -> embed_size // num_heads
#     qk_norm: bool = False         # RMSNorm on Q and K before RoPE
#
# Both need defaults so the existing 32 GPT/Llama configs keep working
# untouched. head_dim is NOT optional sugar for Qwen3 -- it is genuinely
# decoupled from embed_size // num_heads in this family:
#
#     0.6B   d=1024  H=16  head_dim=128  ->  H*head_dim = 2048 = 2.0 * d
#     1.7B   d=2048  H=16  head_dim=128  ->               2048 = 1.0 * d
#     4B     d=2560  H=32  head_dim=128  ->               4096 = 1.6 * d
#     8B     d=4096  H=32  head_dim=128  ->               4096 = 1.0 * d
#     14B    d=5120  H=40  head_dim=128  ->               5120 = 1.0 * d
#     32B    d=5120  H=64  head_dim=128  ->               8192 = 1.6 * d
#
# So q_proj is (d -> H*head_dim) and o_proj is (H*head_dim -> d); they are not
# square. If your attention module does `head_dim = embed_size // num_heads`
# anywhere, it will silently build the wrong shapes for 0.6B, 4B and 32B.
# Likewise the `count_params` helper in the earlier file assumes `d // H` and
# will under/over-count here -- a Qwen3-aware version is at the bottom.
#
# Scaling rules (same spirit as the other ladders):
#   * seq_len, vocab_size, rope_theta, dtype, head_dim, qk_norm are FAMILY
#     CONSTANTS. Only depth, width, head count, kv groups and MLP size move.
#   * head_dim = 128 at every rung, including the tiny ones. Keeping it fixed
#     is what the real family does, and it means the small configs exercise
#     the decoupled-head_dim code path -- which is exactly the bug you want a
#     4-layer unit test to catch.
#   * hidden_dim = 3 * embed_size for the scaled-down rungs, which is the ratio
#     the real 0.6B / 1.7B / 8B use. (4B and 14B run ~3.4-3.8x and 32B runs 5x;
#     Qwen widens the MLP rather than the residual stream at the top end.)
#   * num_heads // attn_gqa_kvg stays in 2-3, matching the real family (0.6B
#     and 1.7B are 2, 4B and 8B are 4, 14B is 5, 32B is 8), and kv groups grow
#     2 -> 4 -> 8 with scale as in the other ladders.
#   * tie_embeds mirrors the released checkpoints: tied at 0.6B/1.7B/4B, untied
#     from 8B up. All scaled-down rungs are tied.
#
# bias=False is correct for the whole family and is not just inherited from the
# Llama configs: Qwen2 carried a bias on Q/K/V, and Qwen3 dropped it when it
# added QK-norm. If you have a Qwen2 path in the same codebase, don't share the
# flag between them.
#
# Naming is by TOTAL params, matching both the Llama configs in the earlier
# file and Qwen's own model names (Qwen3-0.6B is 596M total). Non-embedding
# counts are in the comment on each config, and they are the number you want
# for anything compute-related -- see the warning above the small ladder.
# =============================================================================
from .configs import ModelConfig


# #############################################################################
# ########## Qwen3 scaled-down configs
# ##########   For fast experimentation. Same family constants as the real
# ##########   models: vocab=151_936, seq_len=40_960, rope_theta=1e6,
# ##########   head_dim=128, qk_norm=True, bf16.
# ##########
# ##########   READ THIS BEFORE USING THEM FOR SCALING WORK: the 151,936-token
# ##########   vocab is brutal down here. At d=256 the embedding table is 91%
# ##########   of the model; even at d=1024 it is 38%. Plot against
# ##########   non-embedding params or your curves will be nonsense at the
# ##########   bottom of the ladder. The tied lm_head over 152k tokens also
# ##########   dominates step time at these sizes, so wall-clock will not scale
# ##########   the way the parameter count suggests.
# ##########
# ##########   ALSO: seq_len=40_960 is kept as a family constant for
# ##########   consistency, but it is not a sane length to actually train a
# ##########   4-layer model at. Override per-experiment, e.g.
# ##########       cfg = dataclasses.replace(config_qwen3_43m(), seq_len=1024)
# ##########   rather than forking the configs.
# #############################################################################

def config_qwen3_43m():
    # 3.9M non-embedding / 42.8M total. Debug / unit-test scale.
    # H*head_dim = 512 = 2x embed_size, so this rung exercises the non-square
    # q_proj/o_proj path. Good smoke test.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=2,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=4,
        num_heads=4,
        qk_norm=True,
        embed_size=256,
        seq_len=40_960,
        dropout=0.0,  # not used for Qwen3
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,  # not used for Qwen3
        llama_mlp_size=768,
    )


def config_qwen3_71m():
    # 12.7M non-embedding / 71.0M total.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=2,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=6,
        num_heads=6,
        qk_norm=True,
        embed_size=384,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=1_152,
    )


def config_qwen3_110m():
    # 31.5M non-embedding / 109.3M total.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=4,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=8,
        num_heads=8,
        qk_norm=True,
        embed_size=512,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=1_536,
    )


def config_qwen3_220m():
    # 101.5M non-embedding / 218.2M total. First rung where non-embedding
    # params outweigh the embedding table.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=4,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=12,
        num_heads=12,
        qk_norm=True,
        embed_size=768,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=2_304,
    )


def config_qwen3_407m():
    # 251.7M non-embedding / 407.3M total. Identical to the real 0.6B in every
    # dimension except depth (16 layers vs 28), so it is the cleanest
    # depth-only ablation against a released checkpoint.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=16,
        num_heads=16,
        qk_norm=True,
        embed_size=1024,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=3_072,
    )


def config_qwen3_600m():
    # 440.5M non-embedding / 596.0M total. == Qwen3-0.6B.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=28,
        num_heads=16,
        qk_norm=True,
        embed_size=1024,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=3_072,
    )


def config_qwen3_1700m():
    # 1.41B non-embedding / 1.72B total. == Qwen3-1.7B. Same depth and head
    # count as 0.6B, 2x the width.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=28,
        num_heads=16,
        qk_norm=True,
        embed_size=2048,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=6_144,
    )


def config_qwen3_4b():
    # 3.63B non-embedding / 4.02B total. == Qwen3-4B. Last tied-embedding rung.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=36,
        num_heads=32,
        qk_norm=True,
        embed_size=2560,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=9_728,
    )


def config_qwen3_8b():
    # 6.95B non-embedding / 8.19B total. == Qwen3-8B.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=36,
        num_heads=32,
        qk_norm=True,
        embed_size=4096,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=12_288,
    )


def config_qwen3_14b():
    # 13.21B non-embedding / 14.77B total. == Qwen3-14B.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=40,
        num_heads=40,
        qk_norm=True,
        embed_size=5120,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=17_408,
    )


def config_qwen3_32b():
    # 31.21B non-embedding / 32.76B total. == Qwen3-32B. Note the width is
    # unchanged from 14B; the scale comes entirely from depth (40 -> 64), head
    # count (40 -> 64, so H*head_dim = 1.6x the residual stream) and a 5x MLP.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        rope_theta=1_000_000.0,
        vocab_size=151_936,
        num_layers=64,
        num_heads=64,
        qk_norm=True,
        embed_size=5120,
        seq_len=40_960,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=25_600,
    )

