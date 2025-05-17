import collections
import torch
from typing import  Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
from dlvc.wandb_logger import WandBLogger
from dlvc.dataset.oxfordpets import OxfordPetsCustom

#from dlvc.wandb_logger import WandBLogger

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        '''
        Returns the number of samples in the dataset.
        '''

        pass

#TODO This whole class, without the signatures
class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(self, 
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
                 val_frequency: int = 5):
        '''
        Args and Kwargs:
            model (nn.Module): Deep Network to train
            optimizer (torch.optim): optimizer used to train the network
            loss_fn (torch.nn): loss function used to train the network
            lr_scheduler (torch.optim.lr_scheduler): learning rate scheduler used to train the network
            train_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of training set
            val_metric (dlvc.metrics.SegMetrics): SegMetrics class to get mIoU of validation set
            train_data (dlvc.datasets...): Train dataset
            val_data (dlvc.datasets...): Validation dataset
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

        '''
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
        
        self.subtract_one = isinstance(train_data, OxfordPetsCustom)
        
        self.model.to(self.device)
        
        self.training_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(self.val_data, batch_size=batch_size, shuffle=True)
        self.logger = WandBLogger(model=self.model, enabled=True)
       
        

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.

        epoch_idx (int): Current epoch number
        """
         
        total_loss = 0
        self.train_metric.reset()
        
        for i, data in enumerate(self.training_loader):
            input_data, labels = data
            input_data = input_data.to(self.device)
            labels = labels.squeeze(1)-int(self.subtract_one)
            float_labels = labels.type(torch.LongTensor).to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(input_data)
            if isinstance(outputs, collections.OrderedDict):
                outputs = outputs["out"]
            self.train_metric.update(outputs, labels)
            
            loss = self.loss_fn(outputs, float_labels)
            loss.backward()
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Training epoch {epoch_idx}:\nLoss: {total_loss / (i+1)}\nMean mIoU: {self.train_metric.mIoU()}")
        
        return total_loss, self.train_metric.mIoU()


    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
         
        total_loss = 0
        self.val_metric.reset()
        
        for i, data in enumerate(self.validation_loader):
            input_data, labels = data
            input_data = input_data.to(self.device)
            labels = labels.squeeze(1)-int(self.subtract_one)
            float_labels = labels.type(torch.LongTensor).to(self.device)
            
            outputs = self.model(input_data)
            if isinstance(outputs, collections.OrderedDict):
                outputs = outputs["out"]
            self.val_metric.update(outputs, labels)
            
            loss = self.loss_fn(outputs, float_labels)
                        
            total_loss += loss.item()
        
        print(f"Validation epoch {epoch_idx}:\nLoss: {total_loss/(i+1)}\nMean mIoU: {self.val_metric.mIoU()}")
        
        return total_loss, self.val_metric.mIoU()


    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean IoU on validation data set is higher
        than currently saved best mean IoU or if it is end of training. 
        Depending on the val_frequency parameter, validation is not performed every epoch.
        """
        best = 0
        
        for epoch in range(self.num_epochs):
            
            self.model.train()
            avg_loss, mean_IoU = self._train_epoch(epoch)
            
            if not self.val_frequency or epoch % self.val_frequency == 0:
                self.model.eval()
                val_avg_loss, val_mean_IoU = self._val_epoch(epoch)
                self.logger.log({"validation_avg_loss": val_avg_loss, "validation_mean_IoU": val_mean_IoU}, commit=False)
                if val_mean_IoU > best:
                    self.model.save(self.training_save_dir)
                    best = val_mean_IoU
            self.logger.log({"training_avg_loss": avg_loss, "validation_mean_IoU": val_mean_IoU})
            self.lr_scheduler.step()
        
        self.model.eval()
        val_avg_loss, val_mean_IoU = self._val_epoch(epoch)
        if val_mean_IoU > best:
            self.model.save(self.training_save_dir)           
        
        self.logger.finish()

                





            
            


