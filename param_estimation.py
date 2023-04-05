import fire

'''
Model card: https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md#quantitative-analysis

Example:
65B
python3 param_estimation.py --dim 8192 --n_heads 64 --n_layers 80

7B
python3 param_estimation.py --dim 4096 --n_heads 32 --n_layers 32
'''

def main(
    dim: int = 4096,
    n_heads: int = 32,
    n_layers: int = 32,
    multiple_of: int = 256,
    vocab_size: int = 32000
):
    # Attention
    attention_linear = dim * dim // n_heads * n_heads
    attention_linear *= 4 # each attention block has 4 linear layers

    # Feedforward
    feedforward_hiddendim = int(2 * 4 * dim / 3)
    feedforward_hiddendim = multiple_of * ((feedforward_hiddendim + multiple_of - 1) // multiple_of)
    feedforward_linear = dim * feedforward_hiddendim
    feedforward_linear *= 3 # each feedfoward block has 3 linear layers

    # Transformer block
    # Each transformer blcok includes 1 attention block, 1 feedforward block
    # 2 RMSNorm layers
    transformer_block = feedforward_linear + attention_linear
    transformer_RMSNorm = 2 * dim
    transformer_block += transformer_RMSNorm

    # Transformer model
    # - n_layers of transformer blocks
    # - 1 embedding
    # - 1 linear
    # - 1 RMSNorm
    transformer = n_layers * transformer_block # Transformer blocks
    transformer += vocab_size * dim # embedding
    transformer += vocab_size * dim # linear
    transformer += dim # RMSNorm

    print(f"total {transformer} params ({transformer / 10**9}B)")

if __name__ == "__main__":
    fire.Fire(main)