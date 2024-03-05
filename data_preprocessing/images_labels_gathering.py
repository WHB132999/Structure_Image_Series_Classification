import os
import shutil
import json


absolute_path = os.getcwd()
source_folder = absolute_path.replace('\\', '/') + '/dataset/foundation_images'
target_folder = absolute_path.replace('\\', '/') + '/dataset/final_images_labels/structure_images_wo_special_train'

camera_nums = 4
max_stage_label = 6
structure_num_each_cam = [10, 10, 13, 22]

all_labels = {}
all_structure_labels_file = absolute_path.replace('\\', '/') + '/dataset/final_images_labels/structure_labels_wo_special_train.json'

## Discard the special structures: e.g. cam-01 + structure-9 = 19
special_structure_idx = ['19', '24', '32', '33', '412', '413', '415', '416', '417', '418', '419', '420', '421']  ##['19', '24', '32', '33', '412', '413']


for clean_folder_i in range(1, camera_nums + 1):
    for structure_num_i in range(0, structure_num_each_cam[clean_folder_i - 1]):
        cam_stru_comb = str(clean_folder_i) + str(structure_num_i)
        if cam_stru_comb in special_structure_idx:
            ## These are special structures.
            continue
        else:
            structure_path_i = os.path.join(source_folder,
                                            'clean_cam_0{}'.format(clean_folder_i),
                                            'structure_{}'.format(structure_num_i)).replace('\\', '/')

            files = os.listdir(structure_path_i)

            for file in files:
                if file.endswith('.png'):
                    ## Copy images
                    source_image_path = os.path.join(structure_path_i, file).replace('\\', '/')
                    shutil.copy(source_image_path, target_folder)
                else:
                    assert file.endswith('.json'), "This file should only be json file!"
                    json_path = os.path.join(structure_path_i, file).replace('\\', '/')
                    with open(json_path, 'r') as json_file:

                        json_data = json.load(json_file)
                        all_labels.update(json_data)
            

print(len(all_labels))
with open(all_structure_labels_file, 'w') as clean_json_file:
    json.dump(all_labels, clean_json_file, indent=4)
