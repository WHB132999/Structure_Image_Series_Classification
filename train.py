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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Dataset folders
image_folder_train = './dataset/final_images_labels/structure_images_train'
image_folder_val = './dataset/final_images_labels/structure_images_val'
gt_file_train = './dataset/final_images_labels/structure_labels_train.json'
gt_file_val = './dataset/final_images_labels/structure_labels_val.json'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = StructureDataset(image_folder=image_folder_train, gt_json_file=gt_file_train, transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False)

dataset_val = StructureDataset(image_folder=image_folder_val, gt_json_file=gt_file_val, transform=transform)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

model = build_backbone(model_name='resnet_50', num_classes=dataset_train.num_classes, freeze=True)
model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

num_epochs = 20
best_val_acc = 0.0

training_writer = SummaryWriter('training_logs')

for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    preds_train = []
    labels_train = []

    for inputs_train, labels in dataloader_train:
        inputs_train, labels = inputs_train.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs_train)

        loss = loss_func(outputs, labels)
        loss.backward()

        optimizer.step()

        training_loss += loss.item() * inputs_train.size(0)

        _, predictions_train = torch.max(outputs, 1)

        preds_train.extend(predictions_train.cpu().numpy())
        labels_train.extend(labels.cpu().numpy())


    epoch_loss = training_loss / len(dataset_train)
    acc_train = accuracy_score(labels_train, preds_train)

    model.eval()

    preds_val = []
    labels_val = []

    with torch.no_grad():
        for inputs_val, labels in dataloader_val:
            inputs_val, labels = inputs_val.to(device), labels.to(device)
            
            outputs = model(inputs_val)

            _, predictions_val = torch.max(outputs, 1)

            preds_val.extend(predictions_val.cpu().numpy())
            labels_val.extend(labels.cpu().numpy())

        acc_val = accuracy_score(labels_val, preds_val)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {acc_train:.4f}, Validation Accuracy: {acc_val:.4f}")

        training_writer.add_scalar('Training Loss', epoch_loss, epoch)
        training_writer.add_scalar('Training Accuracy', acc_train, epoch)
        training_writer.add_scalar('Validation Accuracy', acc_val, epoch)

        scheduler.step(acc_val)

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), 'best_model.pth')

training_writer.close()