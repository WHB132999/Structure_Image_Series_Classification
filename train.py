import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataloader import StructureDataset
from backbone import build_backbone

## Only using Resnet50 for classification

## Get the running device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Dataset folders
image_folder_train = './dataset/final_images_labels/structure_image_series_wo_special_train'
image_folder_val = './dataset/final_images_labels/structure_image_series_wo_special_val'
gt_file_train = './dataset/final_images_labels/structure_label_series_wo_special_train.json'
gt_file_val = './dataset/final_images_labels/structure_label_series_wo_special_val.json'

## Transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

## Dataloader for both training and validation
dataset_train = StructureDataset(image_folder=image_folder_train, gt_json_file=gt_file_train, transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False)

dataset_val = StructureDataset(image_folder=image_folder_val, gt_json_file=gt_file_val, transform=transform)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

## Classificaton model
model, _ = build_backbone(model_name='resnet_50', num_classes=dataset_train.num_classes, freeze=False)
model.to(device)

## Loss function, optimizer and scheduler for LR
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

num_epochs = 10
best_val_acc = 0.0

training_writer = SummaryWriter('training_logs')

## Training
for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    preds_train = []
    labels_train = []

    for inputs_train, labels in dataloader_train:
        inputs_train, labels = inputs_train.to(device), labels.to(device)

        ## Clear the gradients from last step
        optimizer.zero_grad()
        outputs, _ = model(inputs_train)

        mono_loss = 0.0

        ## Get the predicted classes
        _, predictions_train = torch.max(outputs, 1)

        ## Insteresting ideas for learning the style of "monotonous increasing GT"
        for idx in range(len(predictions_train)):
            if idx > 0 and int(predictions_train[idx]) < int(predictions_train[idx - 1]):
                mono_loss += 5.0
        
        ## Get total loss
        loss = loss_func(outputs, labels) + mono_loss
        loss.backward()

        ## Update parameters of the model
        optimizer.step()

        training_loss += loss.item() * inputs_train.size(0)

        preds_train.extend(predictions_train.cpu().numpy())
        labels_train.extend(labels.cpu().numpy())

    ## Get training loss and classification accuracy
    epoch_loss = training_loss / len(dataset_train)
    acc_train = accuracy_score(labels_train, preds_train)

    model.eval()

    preds_val = []
    labels_val = []

    ## No gradients for validation
    with torch.no_grad():
        for inputs_val, labels in dataloader_val:
            inputs_val, labels = inputs_val.to(device), labels.to(device)
            
            outputs, _ = model(inputs_val)

            _, predictions_val = torch.max(outputs, 1)

            preds_val.extend(predictions_val.cpu().numpy())
            labels_val.extend(labels.cpu().numpy())

        acc_val = accuracy_score(labels_val, preds_val)

        ## Show the training process
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {acc_train:.4f}, Validation Accuracy: {acc_val:.4f}")

        ## Recording the training and validation results in tensorboard
        training_writer.add_scalar('Training Loss', epoch_loss, epoch)
        training_writer.add_scalar('Training Accuracy', acc_train, epoch)
        training_writer.add_scalar('Validation Accuracy', acc_val, epoch)

        ## Adapt the LR baased on validation accuracy
        scheduler.step(acc_val)

        ## Save the best model's weights
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), 'best_model_49_32_drop_ordered_labels.pth')

training_writer.close()