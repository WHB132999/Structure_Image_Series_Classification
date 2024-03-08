import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

## Statistics of the images for each stage of each structure

## Clean image folder root
clean_image_root = '/dataset/foundation_images'

## Hyperparameters
camera_nums = 4
max_stage_label = 6
structure_num_each_cam = [10, 10, 13, 22]

def statistics_of_structure(folder_path):

    ## Container of the final statistics of each structures
    statistics_of_each_structure = []

    current_path = os.getcwd()
    absolute_path = current_path.replace('\\', '/') + folder_path

    for file_name in os.listdir(absolute_path):

        all_stages_each_structure = []

        if file_name.endswith('.json'):
            file_path = os.path.join(absolute_path, file_name).replace('\\', '/')
            with open(file_path, 'r') as json_file:

                json_data = json.load(json_file)
                all_values = json_data.values()
                for value_i in all_values:

                    ## Get the pure stage number as GT
                    stage_num = int(value_i.split('-')[1].split('_')[0])

                    all_stages_each_structure.append(stage_num)


                ## Verify whether the GT are correct, namely the labels should always increase MONO
                false_label_idx = []
                for i in range(1, len(all_stages_each_structure)):
                    if all_stages_each_structure[i] < all_stages_each_structure[i - 1]:
                        false_label_idx.append(i)
                        all_stages_each_structure[i] = all_stages_each_structure[i - 1]
                
                if false_label_idx:
                    all_stages_each_structure = [] ## set this variable to empty
                    ## This means some labels in this GT-file are wrong
                    ## Then make these labels monotonous increasing
                    for false_idx in false_label_idx:
                        for idx, (key, value) in enumerate(json_data.items()):
                            if idx == false_idx - 1:
                                correct_label = value
                            if idx == false_idx:
                                json_data[key] = correct_label

                    ## Get again all right label sequence for calculating statistics
                    all_values = json_data.values()
                    for value_i in all_values:
                        stage_num = int(value_i.split('-')[1].split('_')[0])
                        all_stages_each_structure.append(stage_num)


                    ## Rebuild the new correct-labels GT-json-file
                    with open(file_path, 'w') as clean_json_file:
                        json.dump(json_data, clean_json_file, indent=4)

                    
                ## Counting the number of each stage of each structure
                for stage_label_i in range(0, max_stage_label + 1):
                    count_stage_i = all_stages_each_structure.count(stage_label_i)
                    statistics_of_each_structure.append(count_stage_i)

    return statistics_of_each_structure


## Container of the final statistics of all structures
statistics_of_all_structures = []

## Load all structures
for clean_folder_i in range(1, camera_nums + 1):
    for structure_num_i in range(0, structure_num_each_cam[clean_folder_i - 1]):
        structure_path_i = os.path.join(clean_image_root,
                                        'clean_cam_0{}'.format(clean_folder_i),
                                        'structure_{}'.format(structure_num_i)).replace('\\', '/')
        print(structure_path_i)
        final_statistics_per_structure = statistics_of_structure(structure_path_i)
        statistics_of_all_structures.append(final_statistics_per_structure)


statistics_array = np.array(statistics_of_all_structures)
print(statistics_array.shape)

## Get the number of all selected images
all_images_nums = statistics_array.sum()

sns.set(style='whitegrid', font_scale=0.7)

## Plot the statistics
plt.figure(figsize=(55, 7))

sns.heatmap(statistics_array, annot=True, fmt='d', cmap='YlGnBu')

## Mark the special structures RED
special_structure_idx = [9, 14, 22, 23, 45, 46]
for idx in special_structure_idx:
    plt.axhspan(ymin=idx, ymax=idx + 1, color='red')

plt.legend(['Red region: Special structures'], loc='upper left')

plt.xlabel('Stage-Number')
plt.ylabel('Structure_Number')

plt.savefig('structure_statistics.png')

plt.show()

