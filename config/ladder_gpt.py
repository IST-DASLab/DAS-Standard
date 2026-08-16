# #############################################################################
# ########## GPT configs
# ##########   dtype=fp32, vocab=50_304, seq_len=1024, head_dim=64, MHSA only.
# ##########   fp32 is inherited from the seed config; you'll want bf16 for
# ##########   anything at or above config_gpt_1_5b to fit/train at speed.
# #############################################################################
from .configs import ModelConfig


def config_gpt_3m():
    # 3.1M non-embedding / 29M total. Debug / unit-test scale.
    return ModelConfig(
        dtype="fp32",
        attn_type="mhsa",
        attn_gqa_kvg=8,  # not used for GPT
        qk_norm=False,
        rope_theta=10_000.0,  # not used for GPT
        vocab_size=50_304,
        num_layers=4,
        num_heads=4,
        embed_size=256,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=True,  # embeddings dominate at this scale; tying is the sane default
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,  # not used for GPT
    )


def config_gpt_11m():
    # 10.6M non-embedding / 50M total.
    return ModelConfig(
        dtype="fp32",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=6,
        num_heads=6,
        embed_size=384,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_25m():
    # 25.2M non-embedding / 77M total.
    return ModelConfig(
        dtype="fp32",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=8,
        num_heads=8,
        embed_size=512,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_49m():
    # 49.2M non-embedding / 114M total.
    return ModelConfig(
        dtype="fp32",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=10,
        num_heads=10,
        embed_size=640,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_85m():
    # 84.9M non-embedding / 163M total. == GPT-2 small. (Your seed config.)
    return ModelConfig(
        dtype="fp32",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=12,
        num_heads=12,
        embed_size=768,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_200m():
    # 201M non-embedding / 305M total. Interpolated between GPT-2 small/medium.
    return ModelConfig(
        dtype="fp32",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=16,
        num_heads=16,
        embed_size=1024,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_300m():
    # 302M non-embedding / 406M total. == GPT-2 medium.
    return ModelConfig(
        dtype="fp32",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=24,
        num_heads=16,
        embed_size=1024,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_710m():
    # 708M non-embedding / 838M total. == GPT-2 large.
    return ModelConfig(
        dtype="fp32",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=36,
        num_heads=20,
        embed_size=1280,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_1500m():
    # 1.47B non-embedding / 1.64B total. == GPT-2 XL.
    # num_heads=25 is the canonical GPT-2 XL value (1600/25 = 64). It's an odd
    # number and will annoy any tensor-parallel sharding; use num_heads=20 with
    # head_dim=80 if you need it to divide cleanly.
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=48,
        num_heads=25,
        embed_size=1600,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_2500m():
    # 2.52B non-embedding / 2.78B total. == GPT-3 "2.7B" shape (head_dim=80).
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=32,
        num_heads=32,
        embed_size=2560,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_6400m():
    # 6.44B non-embedding / 6.86B total. == GPT-3 "6.7B" shape.
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=32,
        num_heads=32,
        embed_size=4096,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )


def config_gpt_12600m():
    # 12.58B non-embedding / 13.10B total. == GPT-3 "13B" shape.
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=50_304,
        num_layers=40,
        num_heads=40,
        embed_size=5120,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=128,
    )