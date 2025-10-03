# quantize_mobilenetv2_multi.py
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import wandb
from quantize_improved import swap_to_quant_modules, freeze_all_quant, evaluate, print_compression

# ------------------- Setup -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------- Argument Parsing -------------------
parser = argparse.ArgumentParser(description='MobileNet-v2 Quantization')
parser.add_argument('--weight_quant_bits', type=int, default=8, help='Weight quantization bits')
parser.add_argument('--activation_quant_bits', type=int, default=8, help='Activation quantization bits')
parser.add_argument('--batchsize', type=int, default=128, help='Batch size for calibration/evaluation')
args = parser.parse_args()

# ------------------- Data -------------------
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)

calibset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
calib_loader = torch.utils.data.DataLoader(calibset, batch_size=2, shuffle=True, num_workers=2)  # small batches for calibration

# ------------------- Model -------------------
model = torchvision.models.mobilenet_v2(weights=None, num_classes=10)
model.load_state_dict(torch.load('./checkpoints/mobilenetv2_cifar10.pth', weights_only=True))
model.to(device)
model.eval()

# ------------------- WandB -------------------
wandb.init(project="mobilenetv2_quant", name="Quantization_MultiConfig", config={
    "weight_bits": args.weight_quant_bits,
    "activation_bits": args.activation_quant_bits,
    "batchsize": args.batchsize
})

# ------------------- FP32 Eval -------------------
fp32_acc = evaluate(model, test_loader, device)
print(f"FP32 Test Accuracy: {fp32_acc:.2f}%")
wandb.log({"FP32_Test_Accuracy": fp32_acc})

# ------------------- Quantization -------------------
print(f"\n=== Quantization: Weights={args.weight_quant_bits} | Activations={args.activation_quant_bits} ===")
swap_to_quant_modules(model, weight_bits=args.weight_quant_bits, act_bits=args.activation_quant_bits, activations_unsigned=True)
model.to(device)

# Calibration: pass a few batches through the model
with torch.no_grad():
    for i, (x, _) in enumerate(calib_loader):
        x = x.to(device)
        _ = model(x)
        if i >= 100:  # calibrate using 100 batches
            break

# Freeze quantization parameters
freeze_all_quant(model)

# ------------------- Quantized Eval -------------------
quant_acc = evaluate(model, test_loader, device)
print(f"Quantized Test Accuracy: {quant_acc:.2f}%")
wandb.log({"Quantized_Test_Accuracy": quant_acc})

# ------------------- Compression -------------------
compression_metrics = print_compression(model, weight_bits=args.weight_quant_bits)

# Log compression metrics to WandB
wandb.log({
    "FP32_model_MB": compression_metrics["FP32_model_MB"],
    "Quant_model_MB": compression_metrics["Quant_model_MB"],
    "Compression_ratio": compression_metrics["Compression_ratio"]
})

wandb.finish()

