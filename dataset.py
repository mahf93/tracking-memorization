import torchvision
from torchvision import datasets, transforms
import torch
import numpy as np
import time


class CIFAR10RandomLabels(datasets.CIFAR10):
    """cifar10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)
        else:
            self.original_targets = self.targets

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets if self.train else self.test_labels)
        self.original_targets = np.array(self.targets if self.train else self.test_labels)
        
        np.random.seed(int(time.time()))
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        labels = [int(x) for x in labels]

        if self.train:
            self.targets = labels
        else:
            self.targets = labels
    def __getitem__(self, index):
        # code from original pytorch cifar10 documentation which I added original target to it
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        original_target = self.original_targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        
        img = torchvision.transforms.ToPILImage()(img)
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, original_target

def get_data_loader(batch_size, train, num_samples=None, corrupt_prob = 0):
    """ get test or train dataloader
    Params
    -----
    batch_size: int
    The size of the batch.
    train: boolean
    If True use the train dataset otherwise use the test dataset.
    num_samples: int
    Default None. The number of samples to use.
    corrupt_prob: float between 0 and 1
    Default 0. The probability of a label being random. 
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if train==True:
        dataset = CIFAR10RandomLabels(root='./data', 
                                       train=True, 
                                       transform=transform_train,
                                       download=True, corrupt_prob = corrupt_prob)
    else:
        dataset = CIFAR10RandomLabels(root='./data', 
                                       train=False, 
                                       transform=transform_test,
                                       download=True)


    if num_samples == None:
        num_samples = len(dataset)
     
    # in case we want to train on part of the training set instead of all
    main_dataset, rest_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[num_samples,len(dataset)-num_samples])

    
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=main_dataset, num_workers = 11,
                                               batch_size=batch_size, 
                                               shuffle=True)

    return data_loader