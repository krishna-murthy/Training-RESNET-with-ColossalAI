from tqdm import tqdm

# For the network
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Optimizer

# For datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# For dataloader
from torch.utils.data import DataLoader

# For distributed training
import colossalai
from colossalai.cluster import DistCoordinator

from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.booster.plugin.dp_plugin_base import DPPluginBase

from colossalai.nn.optimizer import HybridAdam
from colossalai.accelerator import get_accelerator

# Prepare Hyperparameters
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

def get_train_transform_augmentation():
    return transforms.Compose([
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ])

def get_test_transform_augmentation():
    return transforms.ToTensor()

def build_dataloader(batch_size: int, coordinator: DistCoordinator, plugin: DPPluginBase):

    # CIFAR-10 dataset
    data_path = './data'
    with coordinator.priority_execution():
        train_dataset = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=get_train_transform_augmentation()
        )
        test_dataset = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=get_test_transform_augmentation()
        )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Use ColossalAI's data loader plugin
    # Split by batch size on both train and test dataset
    # Set shuffle and drop last to be true for training dataset
    train_dataloader = plugin.prepare_dataloader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = plugin.prepare_dataloader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader

def train(
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        train_dataloader: DataLoader,
        booster: Booster,
        coordinator: DistCoordinator,
):
    model.train()
    with tqdm(train_dataloader, disable=not coordinator.is_master()) as data:
        for images, labels in data:
            images = images.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            labels = labels.to(device='cuda' if torch.cuda.is_available() else 'cpu')
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Print info
            data.set_postfix(loss=loss.item())

@torch.no_grad()
def test(
        model: nn.Module,
        test_dataloader: DataLoader,
        coordinator: DistCoordinator,
):
    model.eval()
    correct = torch.zeros(1, device=get_accelerator().get_current_device())
    total = torch.zeros(1, device=get_accelerator().get_current_device())
    for images, labels in test_dataloader:
        images = images.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        labels = labels.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    dist.all_reduce(correct)
    dist.all_reduce(total)
    accuracy = correct.item() / total.item()
    if coordinator.is_master():
        print(f"Accuracy of the model on the test: {accuracy * 100:.2f} %")

def main():
    # Launch ColossalAI environment with no special config
    # Let world size and rank be default values from torch.distributed
    colossalai.launch_from_torch(config={})
    coordinator = DistCoordinator()
    coordinator.print_on_master('hello world')

    # Get plugin and booster from ColossalAI
    plugin = TorchDDPPlugin()
    booster = Booster(plugin=plugin)

    # Prepare data
    train_dataloader, test_dataloader = build_dataloader(100, coordinator, plugin)

    # Prepare model
    model = torchvision.models.resnet18(num_classes=10)

    # Update the learning rate which was defined earlier scaled with world size value
    global LEARNING_RATE
    LEARNING_RATE = LEARNING_RATE * coordinator.world_size

    # Initialise loss and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = HybridAdam(model.parameters(), lr=LEARNING_RATE)

    # Initialise lr_scheduler
    lr_scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    # Boost all except dataloader with ColossalAI
    model, optimizer, criterion, _, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler
    )

    # Train for num_epochs
    for epoch in range(0, NUM_EPOCHS):
        train(model, optimizer, criterion, train_dataloader, booster, coordinator)
        lr_scheduler.step()

    # Test accuracy
    test(model, test_dataloader, coordinator)

if __name__ == "__main__":
    main()
