import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
from backbone import build_backbone


## Resize the images to same, then transform them to tensor and do normalization to them
def image_preprocess(image_path, resize_shape=(224, 224)):
    preprocess_to_tensor = transforms.Compose([
        transforms.Resize(resize_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
    ])

    input_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_to_tensor(input_image)

    ## Normalize the RGB images for Grad-CAM
    input_image = np.array(input_image) / 255.0

    return input_tensor, input_image


if __name__ == "__main__":
    model = build_backbone(model_name='resnet_50', freeze=False)

    ## Load the trained weights
    best_weights_path = './best_model.pth'
    best_weights_dict = torch.load(best_weights_path)

    model.load_state_dict(best_weights_dict)

    ## Image and GT path for visualization trying
    img_path = './dataset/foundation_images/clean_cam_04/structure_13'
    gt_path = './dataset/structure_13_labels.json'

    gt_labels = []
    with open(gt_path, 'r') as json_file:
        json_data = json.load(json_file)
        all_values = json_data.values()
        for value_i in all_values:
            ## Get pure stage number
            stage_num = int(value_i.split('-')[1].split('_')[0])

            gt_labels.append(stage_num)

    for idx, file_name in enumerate(os.listdir(img_path)):
        file_path = os.path.join(img_path, file_name).replace('\\', '/')

        ## To get the gradients of the feature maps from this convolutional layer
        target_layers = [model.layer4[-1]]

        input_tensor, input_image = image_preprocess(file_path)
        input_tensor = input_tensor.unsqueeze(0)

        ## Construct the cam (class-activation-mapping)
        cam = HiResCAM(model=model, target_layers=target_layers)

        ## If targets is None, means most possible class is highlight
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

        ## Calculate the predicted class
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)

        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

        gt_class = gt_labels[idx]

        ## Visualize the Grad-CAM
        plt.imshow(visualization)

        plt.axis('off')

        plt.title('GT = {}, Pred = {}'.format(gt_class, pred_class))

        plt.savefig('./visualization/Wrong_visual/visual_image_{}'.format(idx))
