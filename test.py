from llama import ModelArgs, Transformer, Tokenizer, LLaMA

tokenizer = Tokenizer(model_path="/home/lsiyuan/spiece.model")

# Number of parameters	dimension	n heads	n layers	Learn rate	Batch size	n tokens
# 7B	4096	32	32	3.0E-04	4M	1T
# 65B   8192	64	80
model_args: ModelArgs = ModelArgs(4096, 32, 32)
# model_args: ModelArgs = ModelArgs(8192, 64, 80)
print("vocab size{}".format(tokenizer.n_words))
model_args.vocab_size = tokenizer.n_words
model = Transformer(model_args)
generator = LLaMA(model, tokenizer)

# n_params = 0
# for param in list(model.parameters()):
#     print("{} elements with element size {}".format(param.nelement(), param.element_size()))
#     n_params += param.nelement()
# print(n_params)