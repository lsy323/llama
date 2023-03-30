import torch
import math

def derive_int8_quant_parameters(w):
    '''
    Create this function for testing purpose, there should be better weight
    quantization algorithms in PyTorch.

    A native function to derive the scaler for int8 weight quantization.
      fp_val = int8_val * 2**(scaler)
    1. Determine the range of the weights
    2. Search the mininum exp (min_exp) such that 2^min_exp * container_range
       covers the range of the weights.
    '''
    weight_range = w.max() - w.min()
    # Find the scaler for the quantized weight. w_float = w_int8 * scaler
    # scaler range: 2^-15, 2^15
    min_exp = -15
    max_exp = 15
    exp = min_exp
    container_range = 2**8
    while exp <= max_exp:
      # 256 is the range of int8
      if math.pow(2, exp) * container_range > weight_range:
          return 2**exp
      exp += 1
    return 2**exp

def quant_weight(w):
    '''
    quant weight 'w' to a int8 tensor
    w: float32 weight
    '''
    container_min = -128
    container_max = 127
    scaler = derive_int8_quant_parameters(w)
    quantized_tensor = w.clone()
    quantized_tensor.detach().apply_(lambda x: round(x / scaler))
    torch.clamp(quantized_tensor, container_min, container_max)
    quantized_tensor = quantized_tensor.to(torch.int8)
    return quantized_tensor, scaler

def dequant_weight(w, scaler):
    '''
    dequant int8 weight 'w' with scler
    w: int8 weights
    fp_w = int8_w * scaler
    '''
    fp_tensor = w.clone()
    fp_tensor = fp_tensor.to(scaler)
    fp_tensor *= scaler
    return fp_tensor

class LinearQuant(torch.nn.Module):
    '''
    Int8 weight-only quantized linaer
    '''
    def __init__(self, in_feature, out_feature, bias=False):
        super().__init__()
        # self.int8_weights = torch.nn.Parameter(torch.zeros((out_feature, in_feature), dtype=torch.int8))
        self.int8_weights = torch.zeros((out_feature, in_feature), dtype=torch.int8)
        self.scaler = torch.tensor(0)
        self.bias = bias
        if bias:
            self.int8_bias = torch.zeros((out_feature), dtype=torch.int8)
            self.bias_scaler = torch.tensor(0)

    def forward(self, x):
        fp_weights = dequant_weight(self.int8_weights, self.scaler)
        activation = torch.matmul(fp_weights, x)
        if self.bias:
            fp_bias = dequant_weight(self.int8_bias, self.bias_scaler)
            activation += fp_bias
        return activation

    def load_weights(self, model):
      '''
      Load weights from a floating point model.
      '''
      # Only load weights for 1 layer now for simplicity.
      self.int8_weights, self.scaler = quant_weight(model.linear.weight)
      dequantized_weight = dequant_weight(self.int8_weights, self.scaler)
