import pickle
from typing import Tuple
import numpy as np
import pickle
import os

from assignment_1_code.datasets.dataset import Subset, ClassificationDataset


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CIFAR10Dataset(ClassificationDataset):
    """
    Custom CIFAR-10 Dataset.
    """

    def __init__(self, fdir: str, subset: Subset, transform=None):
        """
        Initializes the CIFAR-10 dataset.
        """
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

        self.fdir = fdir
        self.subset = subset
        self.transform = transform

        self.images, self.labels = self.load_cifar()

    def load_cifar(self) -> Tuple:
        """
        Loads the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all images from "data_batch_5".
          - The test set contains all images from "test_batch".

        Depending on which subset is selected, the corresponding images and labels are returned.

        Images are loaded in the order they appear in the data files
        and returned as uint8 numpy arrays with shape (32, 32, 3), in RGB channel order.
        """
        
        match self.subset:
            case Subset.TRAINING:
                dict_train_1 = unpickle(os.path.join(self.fdir, "data_batch_1"))
                dict_train_2 = unpickle(os.path.join(self.fdir, "data_batch_2"))
                dict_train_3 = unpickle(os.path.join(self.fdir, "data_batch_3"))
                dict_train_4 = unpickle(os.path.join(self.fdir, "data_batch_4"))
                data = np.concatenate((
                    dict_train_1[b'data'], 
                    dict_train_2[b'data'], 
                    dict_train_3[b'data'], 
                    dict_train_4[b'data']
                    ), 0)
                labels = np.concatenate((
                    dict_train_1[b'labels'], 
                    dict_train_2[b'labels'], 
                    dict_train_3[b'labels'], 
                    dict_train_4[b'labels']
                    ), 0)
            case Subset.VALIDATION:                
                dict_train_5 = unpickle(os.path.join(self.fdir, "data_batch_5"))
                data = dict_train_5[b'data']
                labels = np.array(dict_train_5[b'labels'])
            case Subset.TEST:
                dict_test = unpickle(os.path.join(self.fdir, "test_batch"))
                data = dict_test[b'data']
                labels = np.array(dict_test[b'labels'])
            case _:
                raise ValueError("Invalid subset")

        self.data = data
        self.labels = labels
        
        assert self.data.shape[0] == self.labels.shape[0], "shape mismatch"
        
        return self.data, self.labels


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the idx-th sample in the dataset, which is a tuple,
        consisting of the image and labels.
        Applies transforms if not None.
        Raises IndexError if the index is out of bounds.
        """
        if idx >= len(self) or idx < 0:
            raise IndexError("Index out of bounds")
        
        image = self.data[idx]
        label = self.labels[idx]
        
        image = image.reshape(3, 32, 32).transpose(1, 2, 0)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label

    def num_classes(self) -> int:
        """
        Returns the number of classes.
        """
        return self.labels.max() + 1 - self.labels.min()
