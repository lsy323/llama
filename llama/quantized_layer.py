import torch
import math
import torch.nn.functional as F

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
    return quantized_tensor, torch.Tensor([scaler])

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
    def __init__(self, in_feature, out_feature, bias=False, device=None):
        super().__init__()
        print("in quant linear init")
        # Set requires_grad is necessary as tensor with grad doesn't support integer tensors.
        self.int8_weights = torch.nn.Parameter(torch.randint(-128, 127, (out_feature, in_feature), dtype=torch.int8, device=device), requires_grad=False)
        # self.int8_weights = torch.nn.Parameter(torch.empty(out_feature, in_feature), requires_grad=False)
        # self.int8_weights = torch.zeros((in_feature, out_feature), dtype=torch.int8)
        # self.int8_weights = torch.zeros((out_feature, in_feature), dtype=torch.bfloat16)
        # self.scaler = torch.nn.Parameter(torch.rand(1) / 255, requires_grad=False)
        self.scaler = torch.tensor(1.0 / 128 / math.sqrt(in_feature), device=device)
        # self.scaler = torch.tensor(1.0)
        
        # torch.nn.init.kaiming_uniform_(self.int8_weights, a=math.sqrt(5))
        # # # Default linear layer init:
        # # # https://github.com/pytorch/pytorch/blob/387feaa1312995e33f987175bc27790742272bd4/torch/nn/modules/linear.py#L107
        dummy_fp_weight = torch.empty(out_feature, in_feature)
        torch.nn.init.kaiming_uniform_(dummy_fp_weight, a=math.sqrt(5))
        dummy_fp_weight *= 128 * math.sqrt(in_feature)
        dummy_fp_weight = torch.clamp(dummy_fp_weight, -128, 127)
        self.int8_weights.data = dummy_fp_weight.to(torch.int8)
        # self.load_weights(dummy_fp_weight)

    @torch.no_grad()
    def forward(self, x):
        fp_weights = self.int8_weights * self.scaler
        # x = torch.matmul(x, torch.transpose(fp_weights, 0, 1))
        x = F.linear(x, fp_weights)
        # print(x)
        # x = F.linear(x, self.int8_weights)
        # print(x)
        return x

    def load_weights(self, weight):
      '''
      Load weights from torch.nn.Linear.
      '''
      # Only load weights for 1 layer now for simplicity.
      int8_w, scaler = quant_weight(weight)
      self.int8_weights.copy_(int8_w)
      self.scaler.copy_(scaler)
