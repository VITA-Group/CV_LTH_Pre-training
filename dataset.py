from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST
from torch.utils.data import DataLoader, Subset

__all__ = ['cifar10_dataloaders', 'cifar100_dataloaders', 'svhn_dataloaders', 'fashionmnist_dataloaders']

def cifar10_dataloaders(batch_size=64, data_dir = 'datasets/cifar10'):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def cifar100_dataloaders(batch_size=64, data_dir = 'datasets/cifar100'):

    normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_set = Subset(CIFAR100(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
    val_set = Subset(CIFAR100(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def svhn_dataloaders(batch_size=64, data_dir = 'datasets/svhn'):

    normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True),list(range(68257)))
    val_set = Subset(SVHN(data_dir, split='train', transform=train_transform, download=True),list(range(68257,73257)))
    test_set = SVHN(data_dir, split='test', transform=test_transform, download=True)
            
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

def fashionmnist_dataloaders(batch_size=64, data_dir = 'datasets/fashionmnist'):
    
    normalize = transforms.Normalize(mean=[0.1436], std=[0.1609])
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_set = Subset(FashionMNIST(data_dir, train=True, transform=train_transform, download=True), list(range(55000)))
    val_set = Subset(FashionMNIST(data_dir, train=True, transform=test_transform, download=True), list(range(55000, 60000)))
    test_set = FashionMNIST(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader



