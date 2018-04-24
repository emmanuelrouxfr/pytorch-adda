"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import params
import numpy as np

def get_mnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # dataset and data loader
    mnist_dataset = datasets.MNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)



    if train==True: # the training set has to be edited
        # the nb of samples to train should be fixed to 2000 according to (https://arxiv.org/abs/1702.05464)
        num_train=params.num_samples_in_MNIST
        split = int(num_train)

        indices = list(range(len(mnist_dataset)))
        # to shuffle everytime:
        np.random.shuffle(indices)


        MNIST_idx = indices[:split]
        MNIST_sampler = SubsetRandomSampler(MNIST_idx)

        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset, batch_size=params.batch_size, sampler=MNIST_sampler,
            shuffle=False,
        )

        print("loading data -  MNIST DATA LOADER length is : ")
        print(len(mnist_data_loader))
        print("multiplied by the batch size (" + str(params.batch_size) + ") it should be just enough ( "+ str(params.batch_size) + "x" + str(len(mnist_data_loader)) + " = "+ str(params.batch_size*len(mnist_data_loader)) +") to cover the " + str(params.num_samples_in_MNIST) + "samples drawn from MNIST" )

    else: # the test set can stay complete
        mnist_data_loader = torch.utils.data.DataLoader(
            dataset=mnist_dataset,
            batch_size=params.batch_size,
            shuffle=True)

    return mnist_data_loader
