import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
import statistics

# for wandb users:
from assignment_1_code.wandb_logger import WandBLogger


class BaseTrainer(metaclass=ABCMeta):
    """
    Base class of all Trainers.
    """

    @abstractmethod
    def train(self) -> None:
        """
        Holds training logic.
        """

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float, float]:
        """
        Holds validation logic for one epoch.
        """

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float, float]:
        """
        Holds training logic for one epoch.
        """

        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs: int,
        training_save_dir: Path,
        batch_size: int = 4,
        val_frequency: int = 5,
    ) -> None:
        """
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of training set
            val_metric (dlvc.metrics.Accuracy): Accuracy class to get mAcc and mPCAcc of validation set
            train_data (dlvc.datasets.cifar10.CIFAR10Dataset): Train dataset
            val_data (dlvc.datasets.cifar10.CIFAR10Dataset): Validation dataset
            device (torch.device): cuda or cpu - device used to train the network
            num_epochs (int): number of epochs to train the network
            training_save_dir (Path): the path to the folder where the best model is stored
            batch_size (int): number of samples in one batch
            val_frequency (int): how often validation is conducted during training (if it is 5 then every 5th
                                epoch we evaluate model on validation set)

        What does it do:
            - Stores given variables as instance variables for use in other class methods e.g. self.model = model.
            - Creates data loaders for the train and validation datasets
            - Optionally use weights & biases for tracking metrics and loss: initializer W&B logger

        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        
        self.model.to(self.device)
        
        self.training_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=True)
        self.logger = WandBLogger(model=self.model, enabled=True)
        
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.

        epoch_idx (int): Current epoch number
        """
        
        total_loss = 0
        self.train_metric.reset()
        
        for i, data in enumerate(self.training_loader):
            input_data, labels = data
            input_data = input_data.to(self.device)
            float_labels = labels.type(torch.LongTensor).to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_data)
            self.train_metric.update(outputs, labels)
            
            loss = self.loss_fn(outputs, float_labels)
            loss.backward()
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Training epoch {epoch_idx}:\nLoss: {total_loss / (i+1)}\nMean accuracy: {self.train_metric.accuracy()}")
        
        for k,v in self.train_metric.per_class_accuracy().items():
            print(f"Accuracy for class {k}: {v}")
        
        return total_loss, self.train_metric.accuracy(), statistics.mean(self.train_metric.per_class_accuracy().values())

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        total_loss = 0
        self.val_metric.reset()
        
        for i, data in enumerate(self.validation_loader):
            input_data, labels = data
            input_data = input_data.to(self.device)
            float_labels = labels.type(torch.LongTensor).to(self.device)
            
            outputs = self.model(input_data)
            self.val_metric.update(outputs, labels)
            
            loss = self.loss_fn(outputs, float_labels)
                        
            total_loss += loss.item()
        
        print(f"Validation epoch {epoch_idx}:\nLoss: {total_loss/(i+1)}\nMean accuracy: {self.train_metric.accuracy()}")
        
        for k,v in self.train_metric.per_class_accuracy().items():
            print(f"Accuracy for class {k}: {v}")
        return total_loss, self.train_metric.accuracy(), statistics.mean(self.train_metric.per_class_accuracy().values())

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy.
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        
        best = 0
        
        for epoch in range(self.num_epochs):
            
            self.model.train()
            avg_loss, mean_acc, mean_pc_acc = self._train_epoch(epoch)
            
            if not self.val_frequency or epoch % self.val_frequency == 0:
                self.model.eval()
                val_avg_loss, val_mean_acc, val_mean_pc_acc = self._val_epoch(epoch)
                self.logger.log({"validation_avg_loss": val_avg_loss, "validation_mean_acc": val_mean_acc, "validation_mean_pc_acc": val_mean_pc_acc}, commit=False)
                if val_mean_pc_acc > best:
                    self.model.save(self.training_save_dir/f"best_{type(self.model.net).__name__}.pt")
                    best = val_mean_pc_acc
            self.logger.log({"training_avg_loss": avg_loss, "training_mean_acc": mean_acc, "training_mean_pc_acc": mean_pc_acc})
            self.lr_scheduler.step()
        
        self.model.eval()
        val_avg_loss, val_mean_acc, val_mean_pc_acc = self._val_epoch(epoch)
        if val_mean_pc_acc > best:
            self.model.save(self.training_save_dir/f"best_{type(self.model.net).__name__}.pt")           
        
        self.logger.finish()
            
