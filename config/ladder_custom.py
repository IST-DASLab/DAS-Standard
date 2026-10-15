from .configs import ModelConfig

def config_llama_custom():
    # 973M non-embedding / 1.04B total. == Llama-3.2-1B (head_dim=64,
    # mlp_size=8192 is off-formula; that's the published value).
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=16,
        num_heads=32,
        embed_size=2048,
        seq_len=1024,
        dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=8_192,
    )