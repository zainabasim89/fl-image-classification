"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
from datasets_loaders import create_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from utils.partitioner_helper import get_partitioner
from collections import Counter

pytorch_transforms = Compose([
    Lambda(lambda img: img.convert("RGB")),
    Resize((32, 32)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

dataset = None

def get_dataset():
    global dataset
    # DATASET_PATH = "data/brain-tumor-multimodal-image"
    if dataset is None:
        loader = create_dataset('cifar10', None)
        dataset = loader.load()

    return dataset

def load_data(partition_id: int, num_partitions: int, batch_size: int):

    dataset = get_dataset()
    partitioner = get_partitioner(num_partitions)
    
    partitioner.dataset = dataset["train"]
    partition = partitioner.load_partition(partition_id)

    print(f"Client {partition_id}: Partition size = {len(partition)}")

    # print(Counter(partition["modality"]))
    print(Counter(partition["label"]))
 
    partition = partition.train_test_split(test_size=0.2, seed=42,)

    print(f"Client {partition_id}: train={len(partition["train"])}, test={len(partition["test"])}")

    partition = partition.with_transform(apply_transforms)
    trainloader = DataLoader( partition["train"], batch_size=batch_size, shuffle=True,)
    testloader = DataLoader(partition["test"], batch_size=batch_size, shuffle=False,)

    return trainloader, testloader

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    dataset = get_dataset()
    test_dataset = dataset["test"]
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
