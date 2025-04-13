from abc import ABCMeta, abstractmethod
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """
        if prediction.shape[0] != target.shape[0] or prediction.shape[1] != len(self.classes):
            raise ValueError(f"Wrong dimensions for accuracy update: {prediction.shape} and {target.shape}")
        
        predicted_classes = torch.argmax(prediction, dim=1)
        assert predicted_classes.shape == target.shape
        for pred, t in zip(predicted_classes, target):
            correct = int(pred == t)
            self.n_total += 1
            self.n_matching += correct
            classname = self.classes[t]
            self.total_pred[classname] += 1
            self.correct_pred[classname] += correct
                

    def __str__(self):
        """
        Return a string representation of the performance, accuracy and per class accuracy.
        """

        return f"Correct predictions per class: {self.correct_pred}\nTotal predictions per class: {self.total_pred}\nCorrect predictions: {self.n_matching}\nTotal predictions: {self.n_total}"

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        if self.n_total == 0:
            return 0
        
        return self.n_matching / float(self.n_total)

    def per_class_accuracy(self) -> float:
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        return {classname: self.correct_pred[classname]/float(self.total_pred[classname]) if self.total_pred[classname] else 0 for classname in self.classes}
