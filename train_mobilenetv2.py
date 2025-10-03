import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from mobilenet_v2_model import mobilenet_v2_cifar10
from dataloader import get_cifar10
from utils import evaluate
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize W&B
wandb.init(project="mobilenetv2_quant", name="FP32_training")

if __name__ == "__main__":
    train_loader, test_loader = get_cifar10(batchsize=128)

    model = mobilenet_v2_cifar10()
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=0.07,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=300)

    epochs = 300
    for epoch in range(epochs):
        model.train()
        correct, total, train_loss = 0, 0, 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f} | Train Acc={train_acc:.2f}% | Test Acc={test_acc:.2f}%")

        # Log to W&B
        wandb.log({"epoch": epoch+1,
                   "train_loss": train_loss,
                   "train_acc": train_acc,
                   "test_acc": test_acc,
                   "lr": scheduler.get_last_lr()[0]})

        scheduler.step()

    torch.save(model.state_dict(), "./checkpoints/mobilenetv2_cifar10.pth")

