import os
import json
import shutil
import numpy as np
import pickle
import cv2
from tqdm import tqdm
from roboflow import Roboflow

def display_veriwild_vehicle_types():
    root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1'
    vehicle_info_file_path = os.path.join(root_folder, 'train_test_split', 'vehicle_info.txt')

    with open(vehicle_info_file_path) as file:
        lines = [line.rstrip() for line in file]

    list_vehicle_types = []

    for line in lines:
        current_line = line.split(';')
        current_vehicle_type = current_line[4]

        if current_vehicle_type[0] == ' ':
            current_vehicle_type = current_vehicle_type[1:]
        if not current_vehicle_type in list_vehicle_types:
            list_vehicle_types.append(current_vehicle_type)

    print(list_vehicle_types)

    # classes defined in Veri-wild
    # ['SUV', 'business purpose vehicle/MPV', 'sedan', 'minivan', 'pickup truck', 'HGV/large truck', 'light passenger vehicle', 'large-sized bus', 'small-sized truck', 'bulk lorry/fence truck',

def process_images():
    json_data = {
        "categories": [
            {
                "id": 0,
                "name": "vehicle",
                "supercategory": "none"
            }
        ],
        "images": [
        ],
        "annotations": [
        ]
    }
    #local
    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1'
    # root_images_folder = os.path.join(root_folder, 'images')
    #remote
    root_folder = '/data/veri-wild/veri-wild1/'
    root_images_folder = os.path.join(root_folder, 'images_part01')

    vehicle_info_file_path = os.path.join(root_folder, 'train_test_split', 'vehicle_info.txt')

    with open(vehicle_info_file_path) as file:
        lines = [line.rstrip() for line in file]

    list_vehicle_types = []

    for line in tqdm(lines):
        current_line = line.split(';')
        current_image_path = os.path.join(root_images_folder, current_line[0] + '.jpg')
        current_vehicle_type = current_line[4]

        # for some case of first space ' small-sized truck', remove the additional space
        if current_vehicle_type[0] == ' ':
            current_vehicle_type = current_vehicle_type[1:]

        # find new vehicle type and add it to JSON categories
        if not current_vehicle_type in list_vehicle_types:
            list_vehicle_types.append(current_vehicle_type)
            new_category ={'id': len(json_data['categories']), 'name': current_vehicle_type, 'supercategory': 'vehicle'}
            json_data['categories'].append(new_category)

        # add new image record
        current_image = cv2.imread(current_image_path)
        height, width, _ = current_image.shape
        current_image_id = len(json_data['images'])
        new_image = {'id': current_image_id, 'file_name': current_line[0].split('/')[1], 'height': height, 'width': width}
        json_data['images'].append(new_image)

        # add new image's annotation record

        for cate_idx, cate_item in enumerate(json_data['categories']):
            if cate_item['name'] == current_line[4]:
                current_category_id = cate_item['id']
                break

        new_annotation = {'id': 0, 'image_id': current_image_id, 'category_id': current_category_id,
                          'bbox': [0, 0, width, height], 'area': width * height, 'iscrowd': 0}
        json_data['annotations'].append(new_annotation)


        # if current_vehicle_model in ('MPV', 'SUV', 'Hatchback', 'seadan', 'Minibus', 'Pickup', 'Estate', 'Sport'):

    exported_json_annotation_path = 'exported_annot_json'
    if not os.path.exists(exported_json_annotation_path):
        os.makedirs(exported_json_annotation_path)
    with open(os.path.join(exported_json_annotation_path, 'veriwild1_annotations.json'), 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

    print(list_vehicle_types)

# def generate_yolov8_label():

def process_images_JSON_separated():
    # local
    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'
    # root_images_folder = os.path.join(root_folder, 'images_part01_debug')
    # exported_json_annotation_path = os.path.join(root_images_folder, 'exported_annot_json')

    # remote
    root_folder = '/data/veri-wild/veri-wild1/'
    root_images_folder = os.path.join(root_folder, 'images_part01')
    exported_json_annotation_path = os.path.join(root_images_folder, 'exported_annot_json')

    vehicle_info_file_path = os.path.join(root_folder, 'train_test_split', 'vehicle_info.txt')

    with open(vehicle_info_file_path) as file:
        lines = [line.rstrip() for line in file]

    # list_vehicle_types = []

    for line in tqdm(lines):
        json_data = {
            "categories": [
                {
                    "id": 0,
                    "name": "vehicle",
                    "supercategory": "none"
                },
                {
                    "id": 1,
                    "name": "SUV",
                    "supercategory": "vehicle"
                },
                {
                    "id": 2,
                    "name": "business purpose vehicle/MPV",
                    "supercategory": "vehicle"
                },
                {
                    "id": 3,
                    "name": "sedan",
                    "supercategory": "vehicle"
                },
                {
                    "id": 4,
                    "name": "minivan",
                    "supercategory": "vehicle"
                },
                {
                    "id": 5,
                    "name": "pickup truck",
                    "supercategory": "vehicle"
                },
                {
                    "id": 6,
                    "name": "HGV/large truck",
                    "supercategory": "vehicle"
                },
                {
                    "id": 7,
                    "name": "light passenger vehicle",
                    "supercategory": "vehicle"
                },
                {
                    "id": 8,
                    "name": "large-sized bus",
                    "supercategory": "vehicle"
                },
                {
                    "id": 9,
                    "name": "small-sized truck",
                    "supercategory": "vehicle"
                },
                {
                    "id": 10,
                    "name": "bulk lorry/fence truck",
                    "supercategory": "vehicle"
                },
                {
                    "id": 11,
                    "name": "minibus",
                    "supercategory": "vehicle"
                },
                {
                    "id": 12,
                    "name": "others",
                    "supercategory": "vehicle"
                },
                {
                    "id": 13,
                    "name": "tank car/tanker",
                    "supercategory": "vehicle"
                }
            ],
            "images": [
            ],
            "annotations": [
            ]
        }

        current_line = line.split(';')
        current_line[4] = current_line[4].strip()
        current_image_path = os.path.join(root_images_folder, current_line[0] + '.jpg')
        current_vehicle_type = current_line[4]

        # for some case of first space ' small-sized truck', remove the additional space
        # if current_vehicle_type[0] == ' ':
        #     current_vehicle_type = current_vehicle_type[1:]

        if os.path.isfile(current_image_path) == False:
            continue

        # add new image record
        current_image = cv2.imread(current_image_path)
        height, width, _ = current_image.shape
        # current_image_id = len(json_data['images'])
        current_image_id = int(current_line[0].split('/')[1])
        new_image = {'id': current_image_id, 'file_name': current_line[0].split('/')[1] + '.jpg', 'height': height,
                     'width': width}
        json_data['images'].append(new_image)

        # add new image's annotation record

        for cate_idx, cate_item in enumerate(json_data['categories']):
            if cate_item['name'] == current_line[4]:
                current_category_id = cate_item['id']
                break

        new_annotation = {'id': 0, 'image_id': current_image_id, 'category_id': current_category_id,
                          'bbox': [0, 0, width, height], 'area': width * height, 'iscrowd': 0}
        json_data['annotations'].append(new_annotation)

        if not os.path.exists(exported_json_annotation_path):
            os.makedirs(exported_json_annotation_path)
        with open(os.path.join(exported_json_annotation_path, current_line[0].split('/')[1] + '.json'), 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

def split_train_val_test_dataset():
    # local
    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'
    # root_images_folder = os.path.join(root_folder, 'images_part01_debug')
    # root_yolov8_dataset_folder = os.path.join(root_folder, "yolov8_dataset")
    # train_set_folder = os.path.join(root_yolov8_dataset_folder, "train")
    # valid_set_folder = os.path.join(root_yolov8_dataset_folder, "valid")
    # test_set_folder = os.path.join(root_yolov8_dataset_folder, "test")
    # train_num = 7
    # valid_num = train_num + 2


    # remote
    root_folder = '/data/veri-wild/veri-wild1'
    root_images_folder = os.path.join(root_folder, 'images_part01')
    root_yolov8_dataset_folder = os.path.join(root_folder, "yolov8_dataset")
    train_set_folder = os.path.join(root_yolov8_dataset_folder, "train")
    valid_set_folder = os.path.join(root_yolov8_dataset_folder, "valid")
    test_set_folder = os.path.join(root_yolov8_dataset_folder, "test")
    train_num = 28470
    valid_num = train_num + 8143

    if os.path.exists(root_yolov8_dataset_folder) == False:
        os.makedirs(root_yolov8_dataset_folder)
    if os.path.exists(train_set_folder) == False:
        os.makedirs(train_set_folder)
    if os.path.exists(valid_set_folder) == False:
        os.makedirs(valid_set_folder)
    if os.path.exists(test_set_folder) == False:
        os.makedirs(test_set_folder)

    for subdir, dirs, files in os.walk(root_images_folder):
        for file in files:
            # print(os.path.join(subdir, file))
            source_filepath = os.path.join(subdir, file)

            if source_filepath.endswith(".jpg"):
                num_foler = int(subdir.split("/")[-1])
                if num_foler < train_num:
                    # shutil.copy(source_filepath, os.path.join(train_set_folder, file))
                    shutil.copy(source_filepath, train_set_folder)
                elif num_foler >= train_num and num_foler < valid_num:
                    shutil.copy(source_filepath, valid_set_folder)
                else:
                    shutil.copy(source_filepath, test_set_folder)

                print(source_filepath)

def resize_datasets():
    # prova
    # test_img = cv2.imread(
    #     "/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/vehicle_det_debug-1_STRUCTURE_ROBOFLOW_EXAMPLE/train/images/000003_jpg.rf.5bdd491ed947e19faf71a2b94bb8c575.jpg")
    # orig_img = cv2.imread(
    #     "/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/yolov8_resized_dataset/train/000003.jpg")

    # local
    root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'

    # remote
    # root_folder = '/data/veri-wild/veri-wild1'

    # source images
    root_yolov8_dataset_folder = os.path.join(root_folder, "yolov8_dataset")
    # target resized images
    root_images_resized_folder = os.path.join(root_folder, "yolov8_resized_dataset_debug")
    target_train_set_folder = os.path.join(root_images_resized_folder, "train/images")
    target_valid_set_folder = os.path.join(root_images_resized_folder, "valid/images")
    target_test_set_folder = os.path.join(root_images_resized_folder, "test/images")

    if os.path.exists(root_images_resized_folder) == False:
        os.makedirs(root_images_resized_folder)
    if os.path.exists(target_train_set_folder) == False:
        os.makedirs(target_train_set_folder)
    if os.path.exists(target_valid_set_folder) == False:
        os.makedirs(target_valid_set_folder)
    if os.path.exists(target_test_set_folder) == False:
        os.makedirs(target_test_set_folder)

    for subdir, dirs, files in os.walk(root_yolov8_dataset_folder):
        for file in files:
            source_filepath = os.path.join(subdir, file)
            print(source_filepath)
            if source_filepath.endswith(".jpg"):
                metric = subdir.split("/")[-1]
                if metric == "train":
                        target_filepath = target_train_set_folder
                elif metric == "valid":
                        target_filepath = target_valid_set_folder
                elif metric == "test":
                        target_filepath = target_test_set_folder

                # only python > 3.10
                # match metric:
                #     case "train":
                #         target_filepath = target_train_set_folder
                #     case "valid":
                #         target_filepath = target_valid_set_folder
                #     case "test":
                #         target_filepath = target_test_set_folder

                img = cv2.imread(source_filepath)
                img = cv2.resize(img, (640, 640))
                cv2.imwrite(os.path.join(target_filepath, file), img)

                print(target_filepath)

def generate_yolov8_annotation():
    # local
    root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'
    root_images_resized_folder = os.path.join(root_folder, "yolov8_resized_dataset_debug")

    # remote
    # root_folder = '/data/veri-wild/veri-wild1/'
    # root_images_resized_folder = os.path.join(root_folder, "yolov8_resized_dataset")

    target_train_label_folder = os.path.join(root_images_resized_folder, "train/labels")
    target_valid_label_folder = os.path.join(root_images_resized_folder, "valid/labels")
    target_test_label_folder = os.path.join(root_images_resized_folder, "test/labels")

    if os.path.exists(target_train_label_folder) == False:
        os.makedirs(target_train_label_folder)
    if os.path.exists(target_valid_label_folder) == False:
        os.makedirs(target_valid_label_folder)
    if os.path.exists(target_test_label_folder) == False:
        os.makedirs(target_test_label_folder)

    vehicle_info_file_path = os.path.join(root_folder, 'train_test_split', 'vehicle_info.txt')

    with open(vehicle_info_file_path) as file:
        lines = [line.rstrip() for line in file]

    metrics = ["train", "valid", "test"]

    for metric in metrics:
        metric_images_folder_path = os.path.join(root_images_resized_folder, metric, "images")
        for subdir, dirs, files in os.walk(metric_images_folder_path):
            for file in files:
                image_filename = int(file.split(".")[0])
                line = lines[image_filename - 1]
                # e.g., 00001/000004;14;2018-03-11 10:30:58;Changan;SUV;white
                current_line = line.split(';')
                # [4] type vehicle class
                current_line[4] = current_line[4].strip()
                # current_image_path = os.path.join(root_images_folder, current_line[0] + '.jpg')
                current_vehicle_type = current_line[4]

                # for some case of first space ' small-sized truck', remove the additional space
                if current_vehicle_type[0] == ' ':
                    current_vehicle_type = current_vehicle_type[1:]

                if current_vehicle_type == "SUV":
                    label_current_vehicle_type = 0
                elif current_vehicle_type == "business purpose vehicle/MPV":
                    label_current_vehicle_type = 1
                elif current_vehicle_type == "sedan":
                    label_current_vehicle_type = 2
                elif current_vehicle_type == "minivan":
                    label_current_vehicle_type = 3
                elif current_vehicle_type == "pickup truck":
                    label_current_vehicle_type = 4
                elif current_vehicle_type == "HGV/large truck":
                    label_current_vehicle_type = 5
                elif current_vehicle_type == "light passenger vehicle":
                    label_current_vehicle_type = 6
                elif current_vehicle_type == "large-sized bus":
                    label_current_vehicle_type = 7
                elif current_vehicle_type == "small-sized truck":
                    label_current_vehicle_type = 8
                elif current_vehicle_type == "bulk lorry/fence truck":
                    label_current_vehicle_type = 9
                elif current_vehicle_type == "minibus":
                    label_current_vehicle_type = 10
                elif current_vehicle_type == "others":
                    label_current_vehicle_type = 11
                elif current_vehicle_type == "tank car/tanker":
                    label_current_vehicle_type = 12

                annotation_string = str(label_current_vehicle_type) + " 0.5 0.5 1 1"
                if metric == "train":
                    current_annotation_path = os.path.join(target_train_label_folder, file.split(".")[0] + ".txt")
                elif metric == "valid":
                    current_annotation_path = os.path.join(target_valid_label_folder, file.split(".")[0] + ".txt")
                elif metric == "test":
                    current_annotation_path = os.path.join(target_test_label_folder, file.split(".")[0] + ".txt")

                text_file = open(current_annotation_path, "w")
                text_file.write(annotation_string)
                text_file.close()

                # source_filepath = os.path.join(subdir, file)
                # print(source_filepath)


'''mapping Veri Wild dataset to Blimp vehicle type classes for YOLOV8 det+classif.'''
def generate_mapping_yolov8_det_annotation():
    # local
    root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'
    root_images_resized_folder = os.path.join(root_folder, "yolov8_resized_mapping_dataset_debug")

    # remote
    # root_folder = '/data/veri-wild/veri-wild1/'
    # root_images_resized_folder = os.path.join(root_folder, "yolov8_resized_mapping_dataset")

    target_train_label_folder = os.path.join(root_images_resized_folder, "train/labels")
    target_valid_label_folder = os.path.join(root_images_resized_folder, "valid/labels")
    target_test_label_folder = os.path.join(root_images_resized_folder, "test/labels")

    if os.path.exists(target_train_label_folder) == False:
        os.makedirs(target_train_label_folder)
    if os.path.exists(target_valid_label_folder) == False:
        os.makedirs(target_valid_label_folder)
    if os.path.exists(target_test_label_folder) == False:
        os.makedirs(target_test_label_folder)

    vehicle_info_file_path = os.path.join(root_folder, 'train_test_split', 'vehicle_info.txt')

    with open(vehicle_info_file_path) as file:
        lines = [line.rstrip() for line in file]

    metrics = ["train", "valid", "test"]

    save_annotation = True

    for metric in metrics:
        metric_images_folder_path = os.path.join(root_images_resized_folder, metric, "images")
        for subdir, dirs, files in os.walk(metric_images_folder_path):
            for file in files:
                image_filename = int(file.split(".")[0])
                line = lines[image_filename - 1]
                # e.g., 00001/000004;14;2018-03-11 10:30:58;Changan;SUV;white
                current_line = line.split(';')
                # [4] type vehicle class
                current_line[4] = current_line[4].strip()
                # current_image_path = os.path.join(root_images_folder, current_line[0] + '.jpg')
                current_vehicle_type = current_line[4]

                # for some case of first space ' small-sized truck', remove the additional space
                if current_vehicle_type[0] == ' ':
                    current_vehicle_type = current_vehicle_type[1:]

                if current_vehicle_type == "SUV":
                    save_annotation = True
                    label_current_vehicle_type = 0
                elif current_vehicle_type == "business purpose vehicle/MPV":
                    save_annotation = True
                    label_current_vehicle_type = 1
                elif current_vehicle_type == "sedan":
                    save_annotation = True
                    label_current_vehicle_type = 2
                elif current_vehicle_type == "minivan":
                    save_annotation = True
                    label_current_vehicle_type = 3
                elif current_vehicle_type == "pickup truck":
                    save_annotation = True
                    label_current_vehicle_type = 4
                elif current_vehicle_type == "HGV/large truck":
                    save_annotation = False
                    os.remove(os.path.join(metric_images_folder_path, file))
                    print("Delete: " + os.path.join(metric_images_folder_path, file))
                elif current_vehicle_type == "light passenger vehicle":
                    save_annotation = True
                    label_current_vehicle_type = 5
                elif current_vehicle_type == "large-sized bus":
                    save_annotation = False
                    os.remove(os.path.join(metric_images_folder_path, file))
                    print("Delete: " + os.path.join(metric_images_folder_path, file))
                elif current_vehicle_type == "small-sized truck":
                    save_annotation = False
                    os.remove(os.path.join(metric_images_folder_path, file))
                    print("Delete: " + os.path.join(metric_images_folder_path, file))
                elif current_vehicle_type == "bulk lorry/fence truck":
                    save_annotation = False
                    os.remove(os.path.join(metric_images_folder_path, file))
                    print("Delete: " + os.path.join(metric_images_folder_path, file))
                elif current_vehicle_type == "minibus":
                    label_current_vehicle_type = 3
                elif current_vehicle_type == "others":
                    save_annotation = False
                    os.remove(os.path.join(metric_images_folder_path, file))
                    print("Delete: " + os.path.join(metric_images_folder_path, file))
                elif current_vehicle_type == "tank car/tanker":
                    save_annotation = False
                    os.remove(os.path.join(metric_images_folder_path, file))
                    print("Delete: " + os.path.join(metric_images_folder_path, file))

                if save_annotation == True:
                    annotation_string = str(label_current_vehicle_type) + " 0.5 0.5 1 1"
                    if metric == "train":
                        current_annotation_path = os.path.join(target_train_label_folder, file.split(".")[0] + ".txt")
                    elif metric == "valid":
                        current_annotation_path = os.path.join(target_valid_label_folder, file.split(".")[0] + ".txt")
                    elif metric == "test":
                        current_annotation_path = os.path.join(target_test_label_folder, file.split(".")[0] + ".txt")

                    text_file = open(current_annotation_path, "w")
                    text_file.write(annotation_string)
                    text_file.close()


'''mapping Veri Wild dataset to Blimp vehicle type classes for YOLOV8 det+classif.'''
def generate_mapping_yolov8_classfication_annotation():
    # local
    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'
    # root_images_resized_folder = os.path.join(root_folder, "yolov8_cla_resized_mapping_dataset_debug")

    # remote
    root_folder = '/data/veri-wild/veri-wild1/'
    root_images_resized_folder = os.path.join(root_folder, "yolov8_cla_resized_mapping_dataset")

    vehicle_info_file_path = os.path.join(root_folder, 'train_test_split', 'vehicle_info.txt')

    with open(vehicle_info_file_path) as file:
        lines = [line.rstrip() for line in file]

    metrics = ["train", "valid", "test"]
    str_suv = "SUV"
    str_mpv = "MPV"
    str_sedan = "sedan-fastback"
    str_minibus = "minibus"
    str_pickup = "pickup"
    str_hatchback = "hatchback"

    for metric in metrics:
        metric_images_folder_path = os.path.join(root_images_resized_folder, metric, "images")
        metric_images_SUV_folder_path = os.path.join(metric_images_folder_path, str_suv)
        metric_images_MPV_folder_path = os.path.join(metric_images_folder_path, str_mpv)
        metric_images_sedan_folder_path = os.path.join(metric_images_folder_path, str_sedan)
        metric_images_minibus_folder_path = os.path.join(metric_images_folder_path, str_minibus)
        metric_images_pickup_folder_path = os.path.join(metric_images_folder_path, str_pickup)
        metric_images_hatchback_folder_path = os.path.join(metric_images_folder_path, str_hatchback)

        if not os.path.exists(metric_images_SUV_folder_path):
            os.makedirs(metric_images_SUV_folder_path)
        if not os.path.exists(metric_images_MPV_folder_path):
            os.makedirs(metric_images_MPV_folder_path)
        if not os.path.exists(metric_images_sedan_folder_path):
            os.makedirs(metric_images_sedan_folder_path)
        if not os.path.exists(metric_images_minibus_folder_path):
            os.makedirs(metric_images_minibus_folder_path)
        if not os.path.exists(metric_images_pickup_folder_path):
            os.makedirs(metric_images_pickup_folder_path)
        if not os.path.exists(metric_images_hatchback_folder_path):
            os.makedirs(metric_images_hatchback_folder_path)

    for metric in metrics:
        metric_images_folder_path = os.path.join(root_images_resized_folder, metric, "images")
        for subdir, dirs, files in os.walk(metric_images_folder_path):
            for file in files:
                image_filename = int(file.split(".")[0])
                line = lines[image_filename - 1]
                # e.g., 00001/000004;14;2018-03-11 10:30:58;Changan;SUV;white
                current_line = line.split(';')
                # [4] type vehicle class
                current_line[4] = current_line[4].strip()

                current_vehicle_type = current_line[4]
                current_image_path = os.path.join(metric_images_folder_path, file)

                # for some case of first space ' small-sized truck', remove the additional space
                if current_vehicle_type[0] == ' ':
                    current_vehicle_type = current_vehicle_type[1:]

                if os.path.isfile(current_image_path):
                    if current_vehicle_type == "SUV":
                        label_current_vehicle_type = 0
                        target_image_folder_path = os.path.join(metric_images_folder_path, str_suv, file)
                        os.rename(current_image_path, target_image_folder_path)
                    elif current_vehicle_type == "business purpose vehicle/MPV":
                        label_current_vehicle_type = 1
                        target_image_folder_path = os.path.join(metric_images_folder_path, str_mpv, file)
                        os.rename(current_image_path, target_image_folder_path)
                    elif current_vehicle_type == "sedan":
                        label_current_vehicle_type = 2
                        target_image_folder_path = os.path.join(metric_images_folder_path, str_sedan, file)
                        os.rename(current_image_path, target_image_folder_path)
                    elif current_vehicle_type == "minivan":
                        label_current_vehicle_type = 3
                        target_image_folder_path = os.path.join(metric_images_folder_path, str_minibus, file)
                        os.rename(current_image_path, target_image_folder_path)
                    elif current_vehicle_type == "pickup truck":
                        label_current_vehicle_type = 4
                        target_image_folder_path = os.path.join(metric_images_folder_path, str_pickup, file)
                        os.rename(current_image_path, target_image_folder_path)
                    elif current_vehicle_type == "HGV/large truck":
                        save_annotation = False
                        os.remove(os.path.join(metric_images_folder_path, file))
                        print("Delete: " + os.path.join(metric_images_folder_path, file))
                    elif current_vehicle_type == "light passenger vehicle":
                        save_annotation = True
                        label_current_vehicle_type = 5
                        target_image_folder_path = os.path.join(metric_images_folder_path, str_hatchback, file)
                        os.rename(current_image_path, target_image_folder_path)
                    elif current_vehicle_type == "large-sized bus":
                        save_annotation = False
                        os.remove(os.path.join(metric_images_folder_path, file))
                        print("Delete: " + os.path.join(metric_images_folder_path, file))
                    elif current_vehicle_type == "small-sized truck":
                        save_annotation = False
                        os.remove(os.path.join(metric_images_folder_path, file))
                        print("Delete: " + os.path.join(metric_images_folder_path, file))
                    elif current_vehicle_type == "bulk lorry/fence truck":
                        save_annotation = False
                        os.remove(os.path.join(metric_images_folder_path, file))
                        print("Delete: " + os.path.join(metric_images_folder_path, file))
                    elif current_vehicle_type == "minibus":
                        label_current_vehicle_type = 3
                        target_image_folder_path = os.path.join(metric_images_folder_path, str_minibus, file)
                        os.rename(current_image_path, target_image_folder_path)
                    elif current_vehicle_type == "others":
                        save_annotation = False
                        os.remove(os.path.join(metric_images_folder_path, file))
                        print("Delete: " + os.path.join(metric_images_folder_path, file))
                    elif current_vehicle_type == "tank car/tanker":
                        save_annotation = False
                        os.remove(os.path.join(metric_images_folder_path, file))
                        print("Delete: " + os.path.join(metric_images_folder_path, file))


'''To slow, don't use it '''
def upload_images_from_server():
    # rf = Roboflow(api_key="ivO9dgiYc3AQpHvRWBHC")
    # project = rf.workspace("politecnico-di-milano-iuz9a").project("vehicle_classification-xmkdq")
    # dataset = project.version(1).download("yolov8")

    # creating the Roboflow object
    # obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
    rf = Roboflow(api_key="ivO9dgiYc3AQpHvRWBHC")

    # using the workspace method on the Roboflow object
    workspace = rf.workspace()

    # identifying the project for upload
    project = workspace.project("vehicle_classification-xmkdq")

    # uploading the image to your project
    # project.upload("datasets/1.jpg")

    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/images_part01_debug'
    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/pass_test_new'
    root_folder = '/data/veri-wild/veri-wild1/images_part01'
    annotation_filename = '/data/veri-wild/veri-wild1/veriwild1_annotations.json'
    # annotation_str = open(annotation_filename, "r").read()

    for subdir, dirs, files in tqdm(os.walk(root_folder)):
        for file in files:
            if file.split('.')[-1] == 'jpg':
                print(os.path.join(subdir, file))
                project.upload(os.path.join(subdir, file), annotation_filename)

def upload_images_from_server_JSON_separated():

    # creating the Roboflow object
    rf = Roboflow(api_key="ivO9dgiYc3AQpHvRWBHC")

    # using the workspace method on the Roboflow object
    workspace = rf.workspace()

    # identifying the project for upload
    project = workspace.project("vehicle_det_cl_server")


    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/pass_test_new'

    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/images_part01_debug'

    # root_folder = '/data/veri-wild/veri-wild1/images_part01'
    # annotation_folder = '/data/veri-wild/veri-wild1/images_part01/exported_annot_json'

    # root_folder = '/data/veri-wild/veri-wild1/'

    #local
    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/images_part01_debug'

    # remote
    root_folder = '/data/veri-wild/veri-wild1/images_part01'

    root_images_folder = os.path.join(root_folder, '')
    exported_json_annotation_path = os.path.join(root_images_folder, 'exported_annot_json')

    for subdir, dirs, files in os.walk(root_images_folder):
        for file in files:
            if file.split('.')[-1] == 'jpg':
                print(os.path.join(subdir, file))
                annotation_file_name = file.split('.')[0] + '.json'
                try:
                    project.upload(os.path.join(subdir, file), os.path.join(exported_json_annotation_path, annotation_file_name))
                except Exception:
                    pass

def generate_pytorch_label_format():
    # local
    root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'
    root_images_resized_folder = os.path.join(root_folder, "torchvision_resized_mapping_dataset_debug")

    # remote
    # root_folder = '/data/veri-wild/veri-wild1/'
    # root_images_resized_folder = os.path.join(root_folder, "yolov8_cla_resized_mapping_dataset")

    torchvision_data_path = os.path.join(root_images_resized_folder, "data")
    torchvision_label_path = os.path.join(root_images_resized_folder, "label")
    if os.path.exists(torchvision_label_path) == False:
        os.makedirs(torchvision_label_path)

    vehicle_info_file_path = os.path.join(root_folder, 'train_test_split', 'vehicle_info.txt')

    with open(vehicle_info_file_path) as file:
        lines = [line.rstrip() for line in file]

    metrics = ["train", "valid", "test"]
    # str_suv = "SUV"
    # str_mpv = "MPV"
    # str_sedan = "sedan-fastback"
    # str_minibus = "minibus"
    # str_pickup = "pickup"
    # str_hatchback = "hatchback"

    train_sample_name_list = []
    train_label_list = []
    test_sample_name_list = []
    test_label_list = []
    valid_sample_name_list = []
    valid_label_list = []

    for line in tqdm(lines):
        # e.g., 00001/000004;14;2018-03-11 10:30:58;Changan;SUV;white
        # [4] type vehicle class
        current_line = line.split(';')
        current_line[4] = current_line[4].strip()
        current_image_name = current_line[0].split("/")[1] + '.jpg'
        current_vehicle_type = current_line[4]
        # for some case of first space ' small-sized truck', remove the additional space
        if current_vehicle_type[0] == ' ':
            current_vehicle_type = current_vehicle_type[1:]

        '''search recursively subfolders (train, test, valid) to find whether exists the current_image_name'''
        for root, dirs, files in os.walk(torchvision_data_path):
            if os.path.isfile(os.path.join(root, current_image_name)):
                if root.split("/")[-1] == "train":
                    train_sample_name_list.append(current_image_name)
                    assign_label(current_vehicle_type, train_label_list)
                elif root.split("/")[-1] == "test":
                    test_sample_name_list.append(current_image_name)
                    assign_label(current_vehicle_type, test_label_list)
                elif root.split("/")[-1] == "valid":
                    valid_sample_name_list.append(current_image_name)
                    assign_label(current_vehicle_type, valid_label_list)

    with open(os.path.join(torchvision_label_path, 'train_label.pkl').replace('\\', '/'), 'wb') as f:
        pickle.dump((train_sample_name_list, train_label_list), f)
    with open(os.path.join(torchvision_label_path, 'test_label.pkl').replace('\\', '/'), 'wb') as f:
        pickle.dump((test_sample_name_list, test_label_list), f)
    with open(os.path.join(torchvision_label_path, 'valid_label.pkl').replace('\\', '/'), 'wb') as f:
        pickle.dump((valid_sample_name_list, valid_label_list), f)

    a = 1

def assign_label(current_vehicle_type, label_list):
    if current_vehicle_type == 'SUV':
        label_list.append(0)
    elif current_vehicle_type == 'business purpose vehicle/MPV':
        label_list.append(1)
    elif current_vehicle_type == 'sedan':
        label_list.append(2)
    elif current_vehicle_type == 'minivan':
        label_list.append(3)
    elif current_vehicle_type == 'minibus':
        label_list.append(3)
    elif current_vehicle_type == 'pickup truck':
        label_list.append(4)
    elif current_vehicle_type == 'light passenger vehicle':
        label_list.append(5)


# def generate_pytorch_label_format():
#
#     root_path = "/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1"
#     annotations_json_path = os.path.join(root_path, "veriwild1_annotations.json")
#
#     torchvision_path = os.path.join(root_path, "torchvision_dataset")
#     torchvision_label_path = os.path.join(torchvision_path, "label")
#     torchvision_data_path = os.path.join(torchvision_path, "data")
#     torchvision_train_path = os.path.join(torchvision_data_path, "train")
#     torchvision_valid_path = os.path.join(torchvision_data_path, "valid")
#     torchvision_test_path = os.path.join(torchvision_data_path, "test")
#
#     for foler_name in [torchvision_path, torchvision_label_path, torchvision_data_path, torchvision_train_path, torchvision_valid_path, torchvision_test_path]:
#         if os.path.exists(foler_name) == False:
#             os.makedirs(foler_name)
#
#     with open(annotations_json_path, 'r') as annotations_json:
#         j_data = json.load(annotations_json)
#
#     list_training_label = []
#     list_valid_label = []
#     list_test_label = []
#     for idx, item in enumerate(j_data["annotations"]):
#         if idx <=  train_num:
#             list_training_label.append(item["category_id"])
#         elif idx >  train_num and idx <= valid_num:
#             list_valid_label.append(item["category_id"])
#         else:
#             list_test_label.append(item["category_id"])
#
#     training_label = np.asarray(list_training_label, dtype=np.float32)
#     valid_label = np.asarray(list_valid_label, dtype=np.float32)
#     test_label = np.asarray(list_test_label, dtype=np.float32)
#
#     with open(os.path.join(torchvision_label_path, 'training_label.npy'), 'wb') as f:
#         np.save(f, training_label)
#     with open(os.path.join(torchvision_label_path, 'valid_label.npy'), 'wb') as f:
#         np.save(f, valid_label)
#     with open(os.path.join(torchvision_label_path, 'test_label.npy'), 'wb') as f:
#         np.save(f, test_label)
#
#     a = 3
#

if __name__ == "__main__":
    # process_images()
    # process_images_JSON_separated()
    # upload_images_from_server()
    # upload_images_from_server_JSON_separated()
    # split_train_val_test_dataset()
    # resize_datasets()
    # generate_yolov8_annotation()
    '''Not use yet'''
    # generate_mapping_yolov8_annotation()

    '''Yolo v8 only classification model'''
    # generate_mapping_yolov8_classfication_annotation()

    '''WIP: preparing dataset loader for Pytorch'''
    train_num = 28470
    valid_num = train_num + 8143
    generate_pytorch_label_format()
