from .configs import ModelConfig

def config_llama_1b_custom():
    # 973M non-embedding / 1.04B total. == Llama-3.2-1B (head_dim=64,
    # mlp_size=8192 is off-formula; that's the published value).
    # This is a combination of hyper-params from Llama2-1b and Llama3-1b
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa-flash",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=16,
        num_heads=32,
        embed_size=2048,
        seq_len=1024,
        # dropout=0.0,
        bias=False,
        tie_embeds=True,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=8_192,
    )

def config_llama_4b_custom():
    # 4.15B params
    # config inspired from config_llama3_7200m
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa-flash",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=16, # only 16 layers compared to the 8b one
        num_heads=32,
        embed_size=4096,
        seq_len=1024,
        # dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=16_384,
    )

def config_llama_8b_custom():
    # 8.04B params
    # config inspired from config_llama3_7200m
    return ModelConfig(
        dtype="bf16",
        attn_type="gqa-flash",
        attn_gqa_kvg=8,
        qk_norm=False,
        rope_theta=10_000.0,
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        embed_size=4096,
        seq_len=1024,
        # dropout=0.0,
        bias=False,
        tie_embeds=False,
        gpt_mlp_exp_factor=4,
        llama_mlp_size=16_384,
    )