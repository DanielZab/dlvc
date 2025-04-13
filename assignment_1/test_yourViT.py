## Feel free to change the imports according to your implementation and needs
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
import os

from torchvision.models import resnet18  # change to the model you want to test
from assignment_1_code.models.class_model import DeepClassifier
from assignment_1_code.metrics import Accuracy
from assignment_1_code.datasets.cifar10 import CIFAR10Dataset
from assignment_1_code.datasets.dataset import Subset
from assignment_1_code.models.vit import VisionTransformer


def test(args):

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    fdir = "../cifar-10-python/cifar-10-batches-py"

    test_data = CIFAR10Dataset(fdir, Subset.TEST, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

    device = "cuda"
    num_test_data = len(test_data)

    model = DeepClassifier(VisionTransformer(
                            image_size=32,
                            patch_size=8,
                            num_classes=test_data.num_classes(),
                            dim=16,
                            depth=3,
                            heads=4,
                            mlp_ratio=4.0,
                            qkv_bias=False,
                            qk_scale=None,
                            norm_layer=torch.nn.LayerNorm,
                            pool="cls",
                            ))
    model.load(args.path_to_trained_model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    test_metric = Accuracy(classes=test_data.classes)

    ### Below implement testing loop and print final loss
    ### and metrics to terminal after testing is finished
    test_metric.reset()
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            input_data, labels = batch
            input_data = input_data.to(device)
            float_labels = labels.type(torch.LongTensor).to(device)

            outputs = model(input_data)
            test_metric.update(outputs, labels)
            loss = loss_fn(outputs, float_labels)
            total_loss += loss.item()
        
    print(f"Results:\nLoss: {total_loss/(i + 1)}\nMean accuracy: {test_metric.accuracy()}")
    for k,v in test_metric.per_class_accuracy().items():
        print(f"Accuracy for class {k}: {v}")
        

if __name__ == "__main__":
    ## Feel free to change this part - you do not have to use this argparse and gpu handling
    args = argparse.ArgumentParser(description="Test")
    args.path_to_trained_model = "./saved_models/best_VisionTransformer.pt"

    test(args)
