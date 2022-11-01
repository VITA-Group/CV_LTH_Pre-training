from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, FashionMNIST, Caltech256,Caltech101
from torch.utils.data import DataLoader, Subset
import numpy as np


class FewShotSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.targets = np.array(self.dataset.targets)[indices].tolist()


__all__ = [
    "cifar10_dataloaders",
    "cifar100_dataloaders",
    "svhn_dataloaders",
    "fashionmnist_dataloaders",
    "caltech101_dataloaders",
    "caltech256_dataloaders"
]


def sample_dataset(dataset, per):
    random.seed(1234)
    all_idxs = list()
    for i in range(10):
        idxs = np.where(np.array(dataset["targets"]) == i)[0].tolist()
        all_idxs += random.sample(idxs, 10)

    random.shuffle(all_idxs)

    dataset["targets"] = np.array(dataset["targets"])[all_idxs].tolist()
    dataset["data"] = np.array(dataset["data"])[all_idxs]
    return dataset


# def cifar10_dataloaders(batch_size=64, data_dir = 'datasets/cifar10'):

#     normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize
#     ])

#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])

#     train_set = Subset(CIFAR10(data_dir, train=True, transform=train_transform, download=True), list(range(45000)))
#     val_set = Subset(CIFAR10(data_dir, train=True, transform=test_transform, download=True), list(range(45000, 50000)))
#     test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
#                                 drop_last=True, pin_memory=True)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

#     return train_loader, val_loader, test_loader


def get_balanced_subset(dataset, val_dataset, number_of_samples, val_ratio=0.2):
    number_of_validation_samples = int(number_of_samples / (1 - val_ratio) * val_ratio)
    if number_of_validation_samples + number_of_samples > len(dataset):
        raise ValueError("number of samples is too large")
    unique_labels = np.unique(dataset.targets)
    train_idxs = []
    val_idxs = []
    for label in unique_labels:
        number_of_samples_per_label = int(number_of_samples / len(unique_labels))
        number_of_val_samples_per_label = int(
            number_of_validation_samples / len(unique_labels)
        )
        idxs = np.where(np.array(dataset.targets) == label)[0].tolist()
        train_idxs += idxs[:number_of_samples_per_label]
        val_idxs += idxs[
            number_of_samples_per_label : number_of_samples_per_label
            + number_of_val_samples_per_label
        ]
    dataset_train = FewShotSubset(dataset, train_idxs)
    dataset_val = FewShotSubset(val_dataset, val_idxs)
    return dataset_train, dataset_val


def cifar10_dataloaders(
    batch_size=64,
    data_dir="datasets/cifar10",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if subset_ratio is not None:
        train_set = Subset(
            CIFAR10(data_dir, train=True, transform=train_transform, download=True),
            list(range(int(50000 * subset_ratio * (1 - val_ratio)))),
        )
        val_set = (
            Subset(
                CIFAR10(data_dir, train=True, transform=test_transform, download=True),
                list(
                    range(
                        int(50000 * subset_ratio * (1 - val_ratio)),
                        int(50000 * subset_ratio),
                    )
                ),
            ),
        )

    elif number_of_samples is not None:
        train_set = CIFAR10(
            data_dir, train=True, transform=train_transform, download=True
        )
        val_set = CIFAR10(data_dir, train=True, transform=test_transform, download=True)
        train_set, val_set = get_balanced_subset(
            train_set, val_set, number_of_samples, val_ratio=0.2
        )

    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def cifar100_dataloaders(
    batch_size=64,
    data_dir="datasets/cifar100",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
):

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if subset_ratio is not None:
        train_set = Subset(
            CIFAR100(data_dir, train=True, transform=train_transform, download=True),
            list(range(int(45000 * subset_ratio))),
        )
        val_set = Subset(
            CIFAR100(data_dir, train=True, transform=test_transform, download=True),
            list(range(45000, 50000)),
        )

    elif number_of_samples is not None:
        train_set = CIFAR100(
            data_dir, train=True, transform=train_transform, download=True
        )
        val_set = CIFAR100(
            data_dir, train=True, transform=test_transform, download=True
        )
        train_set, val_set = get_balanced_subset(
            train_set, val_set, number_of_samples, val_ratio=0.2
        )

    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def caltech256_dataloaders(
    batch_size=64,
    data_dir="datasets/caltech256",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
):

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if subset_ratio is not None:
        train_set = Subset(
            Caltech256(data_dir, train=True, transform=train_transform, download=True),
            list(range(int(24486 * subset_ratio))),
        )
        val_set = Subset(
            Caltech256(data_dir, train=True, transform=test_transform, download=True),
            list(range(24486, 30607)),
        )

    elif number_of_samples is not None:
        train_set = Caltech256(
            data_dir, train=True, transform=train_transform, download=True
        )
        val_set = Caltech256(
            data_dir, train=True, transform=test_transform, download=True
        )
        train_set, val_set = get_balanced_subset(
            train_set, val_set, number_of_samples, val_ratio=0.2
        )

    test_set = Caltech256(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader



def caltech101_dataloaders(
    batch_size=64,
    data_dir="datasets/caltech101",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
):

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    if subset_ratio is not None:
        train_set = Subset(
            Caltech101(data_dir, train=True, transform=train_transform, download=True),
            list(range(int(7317  * subset_ratio))),
        )
        val_set = Subset(
            Caltech101(data_dir, train=True, transform=test_transform, download=True),
            list(range(7317, 9146)),
        )

    elif number_of_samples is not None:
        train_set = Caltech101(
            data_dir, train=True, transform=train_transform, download=True
        )
        val_set = Caltech101(
            data_dir, train=True, transform=test_transform, download=True
        )
        train_set, val_set = get_balanced_subset(
            train_set, val_set, number_of_samples, val_ratio=0.2
        )

    test_set = Caltech101(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader

def svhn_dataloaders(
    batch_size=64,
    data_dir="datasets/svhn",
    subset_ratio=None,
    number_of_samples=None,
    val_ratio=0.2,
):

    normalize = transforms.Normalize(
        mean=[0.4377, 0.4438, 0.4728], std=[0.1201, 0.1231, 0.1052]
    )
    train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if subset_ratio is not None:
        train_set = Subset(
            SVHN(data_dir, split="train", transform=train_transform, download=True),
            list(range(int(68257 * subset_ratio))),
        )
        val_set = Subset(
            SVHN(data_dir, split="train", transform=test_transform, download=True),
            list(range(68257, 73257)),
        )
    elif number_of_samples is not None:
        train_set = SVHN(
            data_dir, split="train", transform=train_transform, download=True
        )
        val_set = SVHN(data_dir, split="train", transform=test_transform, download=True)
        train_set, val_set = get_balanced_subset(
            train_set, val_set, number_of_samples, val_ratio=0.2
        )

    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = Subset(
        SVHN(data_dir, split="train", transform=train_transform, download=True),
        list(range(68257)),
    )
    val_set = Subset(
        SVHN(data_dir, split="train", transform=train_transform, download=True),
        list(range(68257, 73257)),
    )
    test_set = SVHN(data_dir, split="test", transform=test_transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def fashionmnist_dataloaders(batch_size=64, data_dir="datasets/fashionmnist"):

    normalize = transforms.Normalize(mean=[0.1436], std=[0.1609])
    train_transform = transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_set = Subset(
        FashionMNIST(data_dir, train=True, transform=train_transform, download=True),
        list(range(55000)),
    )
    val_set = Subset(
        FashionMNIST(data_dir, train=True, transform=test_transform, download=True),
        list(range(55000, 60000)),
    )
    test_set = FashionMNIST(
        data_dir, train=False, transform=test_transform, download=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
