# #############################################################################
# ########## Llama-2 configs
# ##########   dtype=bf16, vocab=32_000, seq_len=4096, rope_theta=10_000.
# ##########   MHSA below ~13B (as in the real family), GQA-8 above.
# #############################################################################
from config.configs import ModelConfig


def config_llama2_35m():
    # 10.6M non-embedding / 35M total.
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=6,  # == num_heads, i.e. MHSA
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=6,
        num_heads=6,
        embed_size=384,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=1_024,
    )


def config_llama2_60m():
    # 27.3M non-embedding / 60M total.
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=8,
        num_heads=8,
        embed_size=512,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=1_536,
    )


def config_llama2_135m():
    # 84.9M non-embedding / 134M total.
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=12,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=12,
        num_heads=12,
        embed_size=768,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=2_048,
    )


def config_llama2_250m():
    # 180M non-embedding / 246M total. First rung with real GQA.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=4,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=16,
        num_heads=16,
        embed_size=1024,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=2_816,
    )


def config_llama2_700m():
    # 604M non-embedding / 702M total. head_dim=128 from here up.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=4,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=24,
        num_heads=12,
        embed_size=1536,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=4_096,
    )


def config_llama2_1100m():
    # 969M non-embedding / 1.10B total. == TinyLlama-1.1B shape (head_dim=64).
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=4,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=22,
        num_heads=32,
        embed_size=2048,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=5_632,
    )


def config_llama2_2800m():
    # 2.62B non-embedding / 2.81B total. Roughly the OpenLLaMA-3B slot.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=26,
        num_heads=24,
        embed_size=3072,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=8_192,
    )


def config_llama2_7b():
    # 6.48B non-embedding / 6.74B total. (Your seed config; rope_theta corrected.)
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=32,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        embed_size=4096,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=11_008,
    )


def config_llama2_13b():
    # 12.69B non-embedding / 13.02B total.
    return ModelConfig(
        dtype="bf16",
        attn_type="mhsa",
        attn_gqa_kvg=40,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=40,
        num_heads=40,
        embed_size=5120,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=13_824,
    )


def config_llama2_34b():
    # 33.22B non-embedding / 33.74B total. == the CodeLlama-34B shape (the
    # Llama-2 34B base model was announced but never released).
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=48,
        num_heads=64,
        embed_size=8192,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=22_016,
    )


def config_llama2_70b():
    # 68.45B non-embedding / 68.98B total. mlp_size uses ffn_multiplier=1.3
    # with multiple_of=4096, which is why it's 28_672 and not 28_416.
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=80,
        num_heads=64,
        embed_size=8192,
        seq_len=4096,
        dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=28_672,
    )