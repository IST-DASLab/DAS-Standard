# #############################################################################
# ########## Llama-3 configs
# ##########   dtype=bf16, vocab=128_256, seq_len=8192, rope_theta=500_000.
# ##########   GQA everywhere. The 128k vocab dominates below ~1B, so every
# ##########   sub-1B rung ties embeddings (as Llama-3.2 1B/3B actually do).
# #############################################################################
from config.configs import ModelConfig


def config_llama3_90m():
    # 27.3M non-embedding / 93M total. Note: 71% of params are the embedding.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=2,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=8,
        num_heads=8,
        embed_size=512,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=1_792,
    )


def config_llama3_200m():
    # 96.7M non-embedding / 195M total.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=4,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=12,
        num_heads=12,
        embed_size=768,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=2_816,
    )


def config_llama3_350m():
    # 218M non-embedding / 349M total.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=4,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=16,
        num_heads=16,
        embed_size=1024,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=3_584,
    )


def config_llama3_1b():
    # 973M non-embedding / 1.24B total. == Llama-3.2-1B (head_dim=64,
    # mlp_size=8192 is off-formula; that's the published value).
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        embed_size=2048,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=8_192,
    )


def config_llama3_3b():
    # 2.82B non-embedding / 3.21B total. == Llama-3.2-3B.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        embed_size=3072,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=8_192,
    )


def config_llama3_8b():
    # 6.98B non-embedding / 8.03B total. == Llama-3-8B. (Your seed config.)
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        embed_size=4096,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=14_336,
    )


def config_llama3_17b():
    # 16.23B non-embedding / 17.55B total. Interpolated -- no real Llama-3 model
    # sits here. Shape follows the family's 1.3x SwiGLU rule at d=5120.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=48,
        num_heads=40,
        embed_size=5120,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=17_920,
    )


def config_llama3_70b():
    # 68.45B non-embedding / 70.55B total. == Llama-3-70B.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=80,
        num_heads=64,
        embed_size=8192,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=28_672,
    )


def config_llama3_405b():
    # 401.65B non-embedding / 405.85B total. == Llama-3.1-405B.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=500_000.0,
        vocab_size=128_256,
        num_layers=126,
        num_heads=128,
        embed_size=16384,
        seq_len=8192,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=53_248,
    )