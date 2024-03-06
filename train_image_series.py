import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataloader import StructureDataset, StructureSeriesDataset
from backbone import build_backbone

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Dataset folders
image_folder_train = './dataset/final_images_labels/structure_image_series_wo_special_train'
image_folder_val = './dataset/final_images_labels/structure_image_series_wo_special_val'
gt_file_train = './dataset/final_images_labels/structure_label_series_wo_special_train.json'
gt_file_val = './dataset/final_images_labels/structure_label_series_wo_special_val.json'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = StructureDataset(image_folder=image_folder_train, gt_json_file=gt_file_train, transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False)

dataset_val = StructureDataset(image_folder=image_folder_val, gt_json_file=gt_file_val, transform=transform)
dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=False)

model, sequence_model = build_backbone(model_name='sequence_model', num_classes=dataset_train.num_classes, freeze=False)
model.to(device)
sequence_model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

optim_seq_model = optim.Adam(sequence_model.parameters(), lr=0.001)
sched_seq_model = lr_scheduler.StepLR(optim_seq_model, step_size=7, gamma=0.1)

num_epochs = 10
best_val_acc = 0.0

training_writer = SummaryWriter('training_logs')

for epoch in range(num_epochs):
    model.train()
    training_loss = 0.0
    preds_train = []
    labels_train = []

    sequence_model.train()
    train_loss_seq_model = 0.0
    preds_train_seq_model = []
    

    for inputs_train, labels in dataloader_train:
        inputs_train, labels = inputs_train.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, last_layer_feats = model(inputs_train)

        optim_seq_model.zero_grad()
        sequence_inputs = last_layer_feats.unsqueeze(0)
        outputs_seq_model = sequence_model(sequence_inputs)
        outputs_seq_model = outputs_seq_model.squeeze()

        loss_classi = loss_func(outputs, labels)
        loss_seq = loss_func(outputs_seq_model, labels)

        loss = loss_classi + loss_seq
        loss.backward()

        optimizer.step()
        optim_seq_model.step()

        training_loss += loss.item() * inputs_train.size(0)
        train_loss_seq_model += loss_seq.item() * inputs_train.size(0)

        _, predictions_train = torch.max(outputs, 1)

        preds_train.extend(predictions_train.cpu().numpy())
        labels_train.extend(labels.cpu().numpy())

        _, predictions_seq_train = torch.max(outputs_seq_model, 1)
        preds_train_seq_model.extend(predictions_seq_train.cpu().numpy())


    epoch_loss = training_loss / len(dataset_train)
    epoch_loss_seq = train_loss_seq_model / len(dataset_train)

    acc_train = accuracy_score(labels_train, preds_train)
    acc_train_seq = accuracy_score(labels_train, preds_train_seq_model)

    model.eval()
    sequence_model.eval()

    preds_val = []
    labels_val = []

    preds_seq_val = []

    with torch.no_grad():
        for inputs_val, labels in dataloader_val:
            inputs_val, labels = inputs_val.to(device), labels.to(device)
            
            outputs, seq_feats = model(inputs_val)

            sequence_inputs = seq_feats.unsqueeze(0)
            outputs_seq_model = sequence_model(sequence_inputs)
            outputs_seq_model = outputs_seq_model.squeeze()

            _, predictions_val = torch.max(outputs, 1)

            _, predictions_seq_val = torch.max(outputs_seq_model, 1)

            preds_val.extend(predictions_val.cpu().numpy())
            labels_val.extend(labels.cpu().numpy())

            preds_seq_val.extend(predictions_seq_val.cpu().numpy())

        acc_val = accuracy_score(labels_val, preds_val)
        acc_val_seq = accuracy_score(labels_val, preds_seq_val)

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Training Seq Loss: {epoch_loss_seq:.4f},"
              f"Training Accuracy: {acc_train:.4f}, Training Seq Accuracy: {acc_train_seq:.4f},"
              f"Validation Accuracy: {acc_val:.4f}, Validation Seq Accuracy: {acc_val_seq:.4f}")

        training_writer.add_scalar('Training Loss', epoch_loss, epoch)
        training_writer.add_scalar('Training Accuracy', acc_train, epoch)
        training_writer.add_scalar('Validation Accuracy', acc_val, epoch)
        training_writer.add_scalar('Training Seq Loss', epoch_loss_seq, epoch)
        training_writer.add_scalar('Training Seq Accuracy', acc_train_seq, epoch)
        training_writer.add_scalar('Validation Seq Accuracy', acc_val_seq, epoch)

        # scheduler.step(acc_val)
        # sched_seq_model.step(acc_val_seq)

        scheduler.step(acc_val)
        sched_seq_model.step()

        if acc_val > best_val_acc:
            best_val_acc = acc_val
            torch.save(model.state_dict(), 'best_model_49_32_nonfreeze_drop.pth')

training_writer.close()