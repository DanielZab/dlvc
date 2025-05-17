# Report DLVC A1

## Questions & Answers

**1. What is the purpose of a loss function? What does the cross-entropy measure? Which criteria must the ground-truth labels and predicted class-scores fulfill to support the cross-entropy loss, and how is this ensured?**

The purpose of a loss function is to evaluate the difference between the ground truth and the prediction of the model.  
The cross-entropy measures the probability of an image belonging to a specific class.  
The ground-truth labels should be one-hot encoded, and the prediciton should be a probability. To ensure a valid probability a softmax is applied.

**2. What is the purpose of the training, validation, and test sets and why do we need all of them?**

To ensure that the model also works on unseen and not just on the training data, the data is split into three parts.  
On the train-set the model is trained. The validation-set is used to tune the hyperparameters. And lastly the test-set is used to evaluate the model on unseen data.  
Additional the validation set does not stay the same, for this the train-set gets split into, so called, folds and on of theses folds become the validation-set and the old validation-set becomes part of the train-set. Doing this multiple times is called cross-validation and the average results is taken for the model.

**3. What are the goals of data augmentation, regularization, and early stopping? How exactly did you use these techniques (hyperparameters, combinations) and what were your results (train, val and test performance)? List all experiments and results, even if they did not work well, and discuss them.**

The purpose of data augmentation, regularization and early stopping is to prevent overfitting.

_DANIEL I NEED U FOR THIS_

**4. What are the key differences between ViTs and CNNs? What are the underlying concepts, respectively? Give the source of your ViT model implementation and point to the specific lines of codes (in your code) where the key concept(s) of the ViT are and explain them.**

_DANIEL I NEED U FOR THIS_
