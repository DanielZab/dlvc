
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
from pathlib import Path
import os

from dlvc.models.segformer import  SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
from copy import deepcopy


def train(dataset, num_epochs):

    train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

    train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])

    if dataset == "oxford":
        train_data = OxfordPetsCustom(root="./oxford", 
                                split="trainval",
                                target_types='segmentation', 
                                transform=train_transform,
                                target_transform=train_transform2,
                                download=True)

        val_data = OxfordPetsCustom(root="./oxford", 
                                split="test",
                                target_types='segmentation', 
                                transform=val_transform,
                                target_transform=val_transform2,
                                download=True)
    if dataset == "city":
        train_data = CityscapesCustom(root="./city", 
                                split="train",
                                mode="fine",
                                target_type='semantic', 
                                transform=train_transform,
                                target_transform=train_transform2)
        val_data = CityscapesCustom(root="./city", 
                                split="val",
                                mode="fine",
                                target_type='semantic', 
                                transform=val_transform,
                                target_transform=val_transform2)


    device = "cuda"

    model = DeepSegmenter(SegFormer(num_classes=len(train_data.classes_seg)))
    # If you are in the fine-tuning phase:
    if dataset == 'oxford':
        ##TODO update the encoder weights of the model with the loaded weights of the pretrained model
        # e.g. load pretrained weights with: state_dict = torch.load("path to model", map_location='cpu')
        state_dict = torch.load(r"E:\Uni OneDrive Big Files\8. Semester\DLVC\repo\assignment_2\saved_models\SegFormer_model.pth", map_location='cpu')
        temp_dict = deepcopy(state_dict)
        for key, value in temp_dict.items():
            del state_dict[key]
            if "encoder" in key:
                state_dict[key.replace("encoder.", "")] = value
                
        model.net.encoder.load_state_dict(state_dict, strict=False)
        
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255) # remember to ignore label value 255 when training with the Cityscapes datset
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2 # for 

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, total_iters=num_epochs)
    
    trainer = ImgSemSegTrainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_metric,
                    val_metric,
                    train_data,
                    val_data,
                    device,
                    num_epochs, 
                    model_save_dir,
                    batch_size=64,
                    val_frequency = val_frequency)
    trainer.train()
    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    num_epochs = 41
    dataset = "oxford"

    train(dataset, num_epochs)