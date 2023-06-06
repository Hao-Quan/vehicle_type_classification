import os
import json
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

def process_images_JSON_separated():
    # local
    root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug'
    root_images_folder = os.path.join(root_folder, 'images_part01_debug')
    # remote
    # root_folder = '/data/veri-wild/veri-wild1/'
    # root_images_folder = os.path.join(root_folder, 'images_part01')

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
        current_image_path = os.path.join(root_images_folder, current_line[0] + '.jpg')
        current_vehicle_type = current_line[4]

        # for some case of first space ' small-sized truck', remove the additional space
        # if current_vehicle_type[0] == ' ':
        #     current_vehicle_type = current_vehicle_type[1:]

        # find new vehicle type and add it to JSON categories
        # if not current_vehicle_type in list_vehicle_types:
        #     list_vehicle_types.append(current_vehicle_type)
        #     new_category = {'id': len(json_data['categories']), 'name': current_vehicle_type,
        #                     'supercategory': 'vehicle'}
        #     json_data['categories'].append(new_category)

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

        # if current_vehicle_model in ('MPV', 'SUV', 'Hatchback', 'seadan', 'Minibus', 'Pickup', 'Estate', 'Sport'):

        exported_json_annotation_path = 'exported_annot_json'
        if not os.path.exists(exported_json_annotation_path):
            os.makedirs(exported_json_annotation_path)
        with open(os.path.join(exported_json_annotation_path, current_line[0].split('/')[1] + '.json'), 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

    # print(list_vehicle_types)

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


if __name__ == "__main__":
    # process_images()
    process_images_JSON_separated()
    # upload_images_from_server()