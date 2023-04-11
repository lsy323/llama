# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.debug.profiler as xp
import torch_xla.distributed.xla_multiprocessing as xmp
import json

from pathlib import Path

from llama import ModelArgs, Transformer, Tokenizer, LLaMA, LinearQuant

from functools import partial
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

def init(
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    use_fsdp: bool = False,
    use_quantized: bool = False,
) -> LLaMA:
    start_time = time.time()
    # FSDP init
    if not use_quantized:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={torch.nn.Linear}) # wrap tranformer block
    else:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={torch.nn.Linear, LinearQuant})
    fsdp_wrap = lambda m: FSDP(
      m,
      compute_dtype=torch.bfloat16,
      fp32_reduce_scatter=False,
      flatten_parameters=False,
      shard_param_on_dim_0=False,
      pin_layout_in_collective_ops=True,
      auto_wrap_policy=auto_wrap_policy,)
    print("Loading")
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    # with open(Path(ckpt_dir) / "params.json", "r") as f:
    #     params = json.loads(f.read())
    params = {"dim": dim,
              "n_layers": n_layers,
              "n_heads": n_heads,
              "use_quantized": use_quantized,
              }
    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)  # TODO: this line puts the model to cuda device
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    model = Transformer(model_args)

    if use_fsdp:
        print("wrapping FSDP")
        model = fsdp_wrap(model)
    else:
        device = xm.xla_device()
        model = model.to(device)
    torch.set_default_tensor_type(torch.FloatTensor)
    # model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    use_fsdp: bool = False,
    use_quantized: bool = False,
):
    server = xp.start_server(9012, only_on_master=False)
    torch.manual_seed(1)
    generator = init(
        tokenizer_path, max_seq_len, max_batch_size, dim, n_layers, n_heads, use_fsdp, use_quantized
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        # "I believe the meaning of life is",
        # "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
#        """Tweet: "I hate it when my phone battery dies."
#Sentiment: Negative
####
#Tweet: "My day has been 👍"
#Sentiment: Positive
####
#Tweet: "This is the link to the article"
#Sentiment: Neutral
####
#Tweet: "This new music video was incredibile"
#Sentiment:""",
#        """Translate English to French:
#
#sea otter => loutre de mer
#
#peppermint => menthe poivrée
#
#plush girafe => girafe peluche
#
#cheese =>""",
    ]
    for _ in range(2):
        with torch.no_grad():
            results = generator.generate(
                prompts, max_gen_len=256, temperature=temperature, top_p=top_p
            )

        for result in results:
            print(result)
            print("\n==================================\n")

def _fn(
    idx,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    use_fsdp: bool = False,
    use_quantized: bool = False,
):
    main(tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, dim, n_layers, n_heads, use_fsdp, use_quantized)

def mp_main(
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    mp: bool = False,
    use_fsdp: bool = False,
    use_quantized: bool = False,
):
    print(f"Use mp: {mp}, Use fsdp {use_fsdp}, Use quantized: {use_quantized}")
    if mp:
        xmp.spawn(_fn, args=(tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, dim, n_layers, n_heads, use_fsdp, use_quantized))
    else:
        main(tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, dim, n_layers, n_heads, use_fsdp, use_quantized)

if __name__ == "__main__":
    # fire.Fire(main)
    fire.Fire(mp_main)
