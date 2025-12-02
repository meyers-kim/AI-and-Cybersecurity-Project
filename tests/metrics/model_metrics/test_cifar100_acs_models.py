import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import functional as TF

from torchvision import datasets, transforms, models as tv_models
from torchvision.utils import save_image

from monai.transforms import RandAffine, RandGaussianNoise, RandGaussianSmooth

import numpy as np
import pandas as pd
from pathlib import Path
import shutil


class TinyCNN100(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeeperCNN100(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_resnet18_100(num_classes: int = 100) -> nn.Module:
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


affine = RandAffine(
    prob=1.0,
    rotate_range=0.1,
    translate_range=(0.05, 0.05),
    scale_range=(0.05, 0.05),
)
noise = RandGaussianNoise(prob=1.0, std=0.05)
blur = RandGaussianSmooth(prob=1.0, sigma_x=(0.5, 1.0))


def apply_monai(x: torch.Tensor) -> torch.Tensor:
    x_t = affine(x)
    x_t = noise(x_t)
    x_t = blur(x_t)
    return x_t


def get_cifar100_loaders(
    data_root: str = "./image_data_cifar100",
    train_subset: int = 2000,
    test_subset: int = 1000,
    batch_size: int = 64,
):
    Path(data_root).mkdir(exist_ok=True)
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform
    )

    train_indices = list(range(min(train_subset, len(train_dataset))))
    test_indices = list(range(min(test_subset, len(test_dataset))))

    train_subset_ds = Subset(train_dataset, train_indices)
    test_subset_ds = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss


def evaluate_acs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_flips: bool = False,
    flip_root: Path | None = None,
    model_name: str = "model",
    max_flips_per_model: int = 10,
):
    """
    Compute:
      - baseline accuracy (on clean images)
      - augmented accuracy (on monai perturbed images)
      - ACS (consistency between baseline and augmented predictions)
      - accuracy drop

    Saves up to `max_flips_per_model` flip examples:
    each saved PNG is a 2-column image [original | augmented], upscaled.
    """
    model.eval()
    all_true: list[int] = []
    all_base: list[int] = []
    all_aug: list[int] = []

    saved = 0
    model_dir: Path | None = None

    if save_flips and flip_root is not None:
        flip_root = Path(flip_root)
        model_dir = flip_root / model_name
        # delete old images for this model
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits_base = model(images)
            images_aug = apply_monai(images)
            logits_aug = model(images_aug)

            y_true = labels.cpu().numpy()
            y_base = logits_base.argmax(dim=1).cpu().numpy()
            y_aug = logits_aug.argmax(dim=1).cpu().numpy()

            all_true.extend(y_true.tolist())
            all_base.extend(y_base.tolist())
            all_aug.extend(y_aug.tolist())

            # save flip examples where prediction changes
            if save_flips and model_dir is not None and saved < max_flips_per_model:
                diff = y_base != y_aug
                idxs = np.where(diff)[0]
                for i in idxs:
                    if saved >= max_flips_per_model:
                        break

                    # original & augmented pair
                    pair = torch.stack([images[i].cpu(), images_aug[i].cpu()], dim=0)

                    # upscale to 256x256
                    upscaled = TF.resize(pair, [256, 256])

                    out_path = (
                        model_dir
                        / f"{model_name}_flip_{saved}_yt{y_true[i]}_b{y_base[i]}_a{y_aug[i]}.png"
                    )
                    save_image(upscaled, out_path, nrow=2)
                    saved += 1

    all_true_arr = np.array(all_true)
    all_base_arr = np.array(all_base)
    all_aug_arr = np.array(all_aug)

    acc_base = (all_true_arr == all_base_arr).mean()
    acc_aug = (all_true_arr == all_aug_arr).mean()
    acs = (all_base_arr == all_aug_arr).mean()
    acc_drop = acc_base - acc_aug
    return acc_base, acc_aug, acs, acc_drop


def test_cifar100_acs_models():
    """
    Compare ACS on CIFAR-100 for several CNN models using MONAI perturbations.
    Saves:
      - numeric results to tests/data/measures/acs_cifar100_models.csv
      - flip images to image_acs_flips_cifar100/<model_name>/...
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_cifar100_loaders()

    models = {
        "TinyCNN100": TinyCNN100(),
        "DeeperCNN100": DeeperCNN100(),
        "ResNet18_100": make_resnet18_100(),
    }

    results: list[dict[str, float | str]] = []
    flip_root = Path("image_acs_flips_cifar100")

    for name, model in models.items():
        print(f"\nTraining {name}")
        model = model.to(device)
        avg_loss = train_one_epoch(model, train_loader, device)
        print(f"{name} - train loss: {avg_loss:.4f}")

        print(f"Evaluating ACS for {name} on CIFAR-100...")
        acc_base, acc_aug, acs, acc_drop = evaluate_acs(
            model,
            test_loader,
            device,
            save_flips=True,
            flip_root=flip_root,
            model_name=name,
            max_flips_per_model=10,
        )
        print(
            f"{name} - acc_base={acc_base:.4f}, "
            f"acc_aug={acc_aug:.4f}, acs={acs:.4f}, acc_drop={acc_drop:.4f}"
        )

        results.append(
            {
                "model": name,
                "acc_base": acc_base,
                "acc_aug": acc_aug,
                "acs": acs,
                "acc_drop": acc_drop,
            }
        )

    measures_dir = Path("tests/data/measures")
    measures_dir.mkdir(parents=True, exist_ok=True)
    out_path = measures_dir / "acs_cifar100_models.csv"

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print("\nSaved results to", out_path)

    assert len(results) == len(models)
