import os
import json
import shutil

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
    # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'

    # remote
    root_folder = '/data/veri-wild/veri-wild1'

    # source images
    root_yolov8_dataset_folder = os.path.join(root_folder, "yolov8_dataset")
    # target resized images
    root_images_resized_folder = os.path.join(root_folder, "yolov8_resized_dataset")
    target_train_set_folder = os.path.join(root_images_resized_folder, "train")
    target_valid_set_folder = os.path.join(root_images_resized_folder, "valid")
    target_test_set_folder = os.path.join(root_images_resized_folder, "test")

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


# def upload_images_1():
#     # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/images_part01_debug'
#     root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/pass_test_new'
#     # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1'
#     # Construct the URL
#     # upload_url = "".join([
#     #     "https://api.roboflow.com/dataset/vehicle_det_debug/upload",
#     #     "?api_key=ivO9dgiYc3AQpHvRWBHC"
#     # ])
#     upload_url = "".join([
#         "https://api.roboflow.com/dataset/vehicle_det_class_api_debug/upload",
#         "?api_key=ivO9dgiYc3AQpHvRWBHC"
#     ])
#
#     # Convert to JPEG Buffer
#     buffered = io.BytesIO()
#
#     for subdir, dirs, files in tqdm(os.walk(root_folder)):
#         for file in files:
#             if file.split('.')[-1] == 'jpg':
#                 print(os.path.join(subdir, file))
#                 # image = Image.open("datasets/1.jpg").convert("RGB")
#                 image = Image.open(os.path.join(subdir,file)).convert("RGB")
#                 image.save(buffered, quality=90, format="JPEG")
#                 m = MultipartEncoder(fields={'file': (file + ".jpg", buffered.getvalue(), "image/jpeg")})
#                 r = requests.post(upload_url, data=m, headers={'Content-Type': m.content_type})
#                 print(r.json())

# def upload_images_example():
#     # creating the Roboflow object
#     # obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
#     rf = Roboflow(api_key="ivO9dgiYc3AQpHvRWBHC")
#
#     # using the workspace method on the Roboflow object
#     workspace = rf.workspace()
#
#     # identifying the project for upload
#     project = workspace.project("vehicle_det_class_api_debug")
#
#     # uploading the image to your project
#     project.upload("datasets/1.jpg")

# def upload_images_2():
#     # creating the Roboflow object
#     # obtaining your API key: https://docs.roboflow.com/rest-api#obtaining-your-api-key
#     rf = Roboflow(api_key="ivO9dgiYc3AQpHvRWBHC")
#
#     # using the workspace method on the Roboflow object
#     workspace = rf.workspace()
#
#     # identifying the project for upload
#     project = workspace.project("vehicle_debug")
#
#     # uploading the image to your project
#     # project.upload("datasets/1.jpg")
#
#     # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/images_part01_debug'
#     root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/pass_test_new'
#     # root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1'
#
#
#     annotation_filename = os.path.join(root_folder, 'veriwild_annotations.json')
#     annotation_str = open(annotation_filename, "r").read()
#
#     for subdir, dirs, files in tqdm(os.walk(root_folder)):
#         for file in files:
#             if file.split('.')[-1] == 'jpg':
#                 print(os.path.join(subdir, file))
#                 project.upload(os.path.join(subdir,file), annotation_filename)
#


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

if __name__ == "__main__":
    # process_images()
    # process_images_JSON_separated()
    # upload_images_from_server()
    # upload_images_from_server_JSON_separated()
    # split_train_val_test_dataset()
    resize_datasets()