from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

#TODO This whole class, without the signatures
class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.correct_pred = torch.zeros((len(self.classes),)).int()
        self.total_pred = torch.zeros((len(self.classes),)).int()
        self.false_positives = torch.zeros((len(self.classes),)).int()



    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''

        if (len(prediction.shape) != 4 or
            prediction.shape[0] != target.shape[0] or 
            prediction.shape[1] != len(self.classes) or 
            prediction.shape[2] != target.shape[1] or 
            prediction.shape[3] != target.shape[2]):
            raise ValueError(f"Wrong dimensions for accuracy update: {prediction.shape} and {target.shape}")

        predicted_classes = torch.argmax(prediction, dim=1)
        temp_target = target[target != 255].flatten().cpu()
        predicted_classes = predicted_classes[target != 255].flatten().cpu()
        target = temp_target
        assert predicted_classes.shape == target.shape
        self.correct_pred.scatter_reduce_(-1, predicted_classes,(predicted_classes == target).int(), reduce='sum')
        self.total_pred.scatter_reduce_(-1, target, torch.ones_like(target).int(), reduce='sum')
        self.false_positives.scatter_reduce_(-1, predicted_classes, (predicted_classes != target).int(), reduce='sum')


    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIoU: {self.mIoU()}"
          

    
    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        return (self.correct_pred / (self.total_pred + self.false_positives)).mean()





