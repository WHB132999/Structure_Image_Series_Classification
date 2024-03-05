import csv
import os
import shutil
import json


def labels_extraction(path_to_labels_file):

    ## Using Dict to avoid the same labels, clean the labels
    extracted_labels = {}

    with open(path_to_labels_file, 'r', newline='', encoding="utf-8") as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        for file_label in csv_reader:

            split_file_label = file_label[0].split(';')

            image_name = split_file_label[0].replace(':', '_')
            gt_label = split_file_label[1]

            extracted_labels[image_name] = gt_label
    
    return extracted_labels


csv_file_path = '../dataset/stage_labels.csv'
foundation_labels = labels_extraction(csv_file_path)

## Gathering the labels in a list and Found some same labels
# seen_files = set()
# for i in foundation_labels:
#     if i in seen_files:
#         print("Seen file names: {}".format(i))
#     else:
#         seen_files.add(i)


foundation_images_path = '../dataset/foundation_images'

## cam_1: 10 structures, cam_2: 10 structures, cam_3: 13 structures, cam_4: 22 structures
structure_nums = [10, 10, 13, 22]


## 4 cameras -> range(1, 5)
for i in range(1, 5):

    ## Build the new image folders for the filtered images
    new_folder_name = 'clean_cam_0{}'.format(i)
    new_folder_path = os.path.join(foundation_images_path, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    for num in range(structure_nums[i - 1]):
        structure_folder_name = 'structure_{}'.format(num)
        structure_path = os.path.join(new_folder_path, structure_folder_name)
        os.makedirs(structure_path, exist_ok=True)

    path_i = foundation_images_path + '/cam_0{}'.format(i)

    ## Build dists for gt_labels for each camera
    dict_list = [{} for _ in range(structure_nums[i - 1])]

    for image_i in os.listdir(path_i):

        if image_i in foundation_labels:

            ## Source image path
            source_image_path = os.path.join(path_i, image_i).replace('\\', '/')

            structure_num = image_i.split(';')[0].split('_')[-1].split('.')[0]

            ## Build the name of goal_folder
            structure_goal_path = os.path.join(new_folder_path, 'structure_{}'.format(structure_num), image_i).replace('\\', '/')

            ## Copy the filtered image from source folder to the destination folder
            shutil.copyfile(source_image_path, structure_goal_path)

            ## Add the gt_labels of each selected image to the gt_dict
            dict_list[int(structure_num)][image_i] = foundation_labels[image_i]
        else:
            continue

    for idx, gt_dict in enumerate(dict_list):
        json_file_name = 'structure_{}_labels.json'.format(idx)
        sub_folder_name = 'structure_{}'.format(idx)
        json_file_path = os.path.join(new_folder_path, sub_folder_name, json_file_name)
        with open(json_file_path, 'w') as json_file:
            json.dump(gt_dict, json_file, indent=4)



def labels_cleaning(extracted_labels):
    camera_1 = []
    camera_2 = []
    camera_3 = []
    camera_4 = []





