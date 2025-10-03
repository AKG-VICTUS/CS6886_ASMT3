import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- Quantization helpers -------------------
def quantize_tensor(x, scale, zp, qmin, qmax):
    # Make sure scale and zp are on the same device as x
    scale = scale.to(x.device)
    zp = torch.tensor(zp, device=x.device)
    q = torch.round(x / scale + zp)
    return q.clamp(qmin, qmax)

def dequantize_tensor(q, scale, zp):
    scale = scale.to(q.device)
    zp = torch.tensor(zp, device=q.device)
    return (q - zp) * scale

# ------------------- FakeQuant Activation -------------------
class ActFakeQuant(nn.Module):
    def __init__(self, n_bits=8, unsigned=True):
        super().__init__()
        self.n_bits = n_bits
        self.unsigned = unsigned
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin, self.qmax = None, None

    @torch.no_grad()
    def observe(self, x):
        self.min_val = torch.minimum(self.min_val, x.min())
        self.max_val = torch.maximum(self.max_val, x.max())

    @torch.no_grad()
    def freeze(self):
        qmin, qmax = (0, (1 << self.n_bits) - 1) if self.unsigned else (-((1 << (self.n_bits - 1)) - 1), (1 << (self.n_bits - 1)) - 1)
        scale = (self.max_val - self.min_val) / (qmax - qmin + 1e-12)
        zp = torch.round(-self.min_val / scale).clamp(qmin, qmax)
        self.scale.copy_(scale)
        self.zp.copy_(zp)
        self.qmin, self.qmax = qmin, qmax
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            self.observe(x)
            return x
        q = quantize_tensor(x, self.scale, self.zp, self.qmin, self.qmax)
        return dequantize_tensor(q, self.scale, self.zp)

# ------------------- QuantConv / QuantLinear -------------------
class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        w_min, w_max = self.weight.min(), self.weight.max()
        self.qmin, self.qmax = -((1 << (self.weight_bits - 1)) - 1), (1 << (self.weight_bits - 1)) - 1
        self.scale.copy_(torch.tensor(max(abs(w_min), abs(w_max)) / self.qmax))
        self.zp.copy_(torch.tensor(0.0))
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        w_q = quantize_tensor(self.weight, self.scale, self.zp, self.qmin, self.qmax)
        w_dq = dequantize_tensor(w_q, self.scale, self.zp)
        return F.conv2d(x, w_dq, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    def __init__(self, *args, weight_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("zp", torch.tensor(0.0))
        self.frozen = False
        self.qmin = None
        self.qmax = None

    @torch.no_grad()
    def freeze(self):
        w_min, w_max = self.weight.min(), self.weight.max()
        self.qmin, self.qmax = -((1 << (self.weight_bits - 1)) - 1), (1 << (self.weight_bits - 1)) - 1
        self.scale.copy_(torch.tensor(max(abs(w_min), abs(w_max)) / self.qmax))
        self.zp.copy_(torch.tensor(0.0))
        self.frozen = True

    def forward(self, x):
        if not self.frozen:
            return F.linear(x, self.weight, self.bias)
        w_q = quantize_tensor(self.weight, self.scale, self.zp, self.qmin, self.qmax)
        w_dq = dequantize_tensor(w_q, self.scale, self.zp)
        return F.linear(x, w_dq, self.bias)

# ------------------- Model Surgery -------------------
def swap_to_quant_modules(model, weight_bits=8, act_bits=8, activations_unsigned=True):
    for name, m in list(model.named_children()):
        swap_to_quant_modules(m, weight_bits, act_bits, activations_unsigned)

        if isinstance(m, nn.Conv2d):
            q = QuantConv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                stride=m.stride, padding=m.padding, dilation=m.dilation,
                groups=m.groups, bias=(m.bias is not None),
                weight_bits=weight_bits
            )
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.Linear):
            q = QuantLinear(m.in_features, m.out_features, bias=(m.bias is not None), weight_bits=weight_bits)
            q.weight.data.copy_(m.weight.data)
            if m.bias is not None:
                q.bias.data.copy_(m.bias.data)
            setattr(model, name, q)

        elif isinstance(m, nn.ReLU):
            seq = nn.Sequential(nn.ReLU(inplace=getattr(m, "inplace", False)),
                                ActFakeQuant(n_bits=act_bits, unsigned=activations_unsigned))
            setattr(model, name, seq)

def freeze_all_quant(model):
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            mod.freeze()
        if isinstance(mod, ActFakeQuant):
            mod.freeze()

# ------------------- Evaluation -------------------
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# ------------------- Compression -------------------
def model_size_bytes_fp32(model):
    return sum(p.numel() * 4 for p in model.parameters())

def model_size_bytes_quant(model, weight_bits=8):
    total = 0
    for name, p in model.named_parameters():
        if "weight" in name:
            total += p.numel() * weight_bits // 8
        else:
            total += p.numel() * 4
    return total

def print_compression(model, weight_bits=8):
    fp32 = model_size_bytes_fp32(model)
    quant = model_size_bytes_quant(model, weight_bits)
    ratio = fp32 / max(quant, 1)
    print(f"FP32 model: {fp32/1024/1024:.2f} MB, Quant: {quant/1024/1024:.2f} MB, Ratio: {ratio:.2f}x")
    return {"FP32_model_MB": fp32/1024/1024, "Quant_model_MB": quant/1024/1024, "Compression_ratio": ratio}

