import torch
import torchvision

def load_datasets(batch_size=32, degrees_rotation=20, train_size=0.97, background_samples=True, extra=False):
    """
    Loads datasets for model training.

    Args:
        batch_size (int): the batch size to use.
        degrees_rotation (int): the maximum amount of rotation to apply to samples in the RandomRotation() transform
        train_size (float): the training percentage for the trian/val split
        background_samples (bool): a boolean specifying if background samples from the DTD dataset should be used
        extra (bool): a boolean specifying if the extra training data from SVHN should be used
    
    Returns:
        The train loader, validation loader, and test loader for the datasets.
    """
    # Compose transformations
    transforms = torchvision.transforms.Compose([torchvision.transforms.RandomRotation(degrees_rotation),
                                                 torchvision.transforms.ToTensor()])

    background_transforms = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((32, 32), scale=(0.01, 0.33), ratio=(0.25, 1.75)),
                                                            torchvision.transforms.ToTensor()])

    # Load train, test, and background samples and extra data
    trainset = torchvision.datasets.SVHN(
        root='./dataset', split='train', download=True, transform=transforms)

    if background_samples:
        dtd_trainset = torchvision.datasets.DTD(
            root='./dataset', split='train', transform=background_transforms, target_transform=lambda x: 10, download=True)
        trainset = torch.utils.data.ConcatDataset([trainset, dtd_trainset])

    if extra:
        extraset = torchvision.datasets.SVHN(
            root='./dataset', split='extra', download=True, transform=transforms)
        trainset = torch.utils.data.ConcatDataset([trainset, extraset])

    testset = torchvision.datasets.SVHN(
        root='./dataset', split='test', download=True, transform=transforms)

    # Split val set from training data
    dataset_size = len(trainset)
    train_size = int(train_size * dataset_size)
    val_size = dataset_size - train_size
    trainset, valset = torch.utils.data.random_split(
        trainset, [train_size, val_size])

    # Create train, val, and test loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader
