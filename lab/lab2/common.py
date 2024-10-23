from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
import torch
from thop import profile
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_cifar10_loaders(
    batch_size,
    transform_train=None,
    transform_test=None,
    normalization=[],
    augmentation=[],
    val_size=5000,
    verbose=True,
):
    if transform_train is None:
        transform_train = transforms.Compose(
            [
                *augmentation,
                transforms.ToTensor(),
                *normalization,
            ]
        )
    if transform_test is None:
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                *normalization,
            ]
        )
    if verbose:
        print(f"transform_train: {transform_train}")
        print(f"transform_test: {transform_test}")

    torch.manual_seed(43)

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    trainset, valset = random_split(trainset, [len(trainset) - val_size, val_size])
    if verbose:
        print("train length: ", len(trainset))
        print("val length: ", len(valset))
        print("test length: ", len(testset))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader


def show_model_info(model, input_size):
    flops, params = profile(model, inputs=(torch.randn(1, *input_size).cuda(),))
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")

    summary(model, input_size, device="cuda")


def train_one_epoch(model, loader, criterion, optimizer, device="cuda"):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device="cuda"):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = val_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def train(
    model,
    trainloader,
    valloader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=10,
    save_path="best.pt",
    existed="keep_both",
    early_stop_patience=0,
    device="cuda",
):
    model.train().to(device)
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    early_stop_count = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, valloader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f'Epoch {epoch+1}/{epochs} (lr={optimizer.param_groups[0]["lr"]:.1e})',
            end=", ",
        )
        print(f"train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}", end=", ")
        print(f"train_acc: {train_acc:.3f}, val_acc: {val_acc:.3f}")

        if scheduler:
            scheduler.step()

        if val_acc == max(val_accs):
            save_model(model.eval(), save_path, existed=existed)

        if val_loss > min(val_losses):
            early_stop_count += 1
        else:
            early_stop_count = 0
        if early_stop_count >= early_stop_patience:
            print(f"Early stopping at {epoch+1} epoch")
            break

    return train_losses, train_accs, val_losses, val_accs


def preprocess_filename(filename: str, existed: str = "keep_both") -> str:
    if existed == "overwrite":
        pass
    elif existed == "keep_both":
        base, ext = os.path.splitext(filename)
        cnt = 1
        while os.path.exists(filename):
            filename = f"{base}-{cnt}{ext}"
            cnt += 1
    elif existed == "raise" and os.path.exists(filename):
        raise FileExistsError(f"{filename} already exists.")
    else:
        raise ValueError(f"Unknown value for 'existed': {existed}")
    return os.path.abspath(filename)


def save_model(
    model, filename: str, verbose: bool = True, existed: str = "keep_both"
) -> None:
    filename = preprocess_filename(filename, existed)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(model.state_dict(), filename)
    if verbose:
        print(f"Model saved at {filename} ({os.path.getsize(filename) / 1e6} MB)")


def load_model(model, filename: str, verbose: bool = True) -> None:
    state = torch.load(filename)
    if "total_ops" in state:
        state.pop("total_ops")
    if "total_params" in state:
        state.pop("total_params")
    model.load_state_dict(state)
    if verbose:
        print(f"Model loaded from {filename} ({os.path.getsize(filename) / 1e6} MB)")


def plot_loss_accuracy(
    train_loss, train_acc, val_loss, val_acc, filename="loss_accuracy.png"
):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.plot(train_loss, color="tab:blue")
    ax1.plot(val_loss, color="tab:red")
    ax1.legend(["Training", "Validation"])
    ax1.set_title("Loss")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.plot(train_acc, color="tab:blue")
    ax2.plot(val_acc, color="tab:red")
    ax2.legend(["Training", "Validation"])
    ax2.set_title("Accuracy")

    fig.tight_layout()
    if filename:
        filename = preprocess_filename(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        print(f"Plot saved at {filename}")
    plt.show()


@dataclass
class HyperParameter:
    batch_size: int
    lr: float
    epochs: int
