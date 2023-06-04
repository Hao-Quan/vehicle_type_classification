import os
import json
import cv2

# def preprare_header():
#     json_data = {
#         "categories": [
#             {
#                 "id": 0,
#                 "name": "vehicle",
#                 "supercategory": "none"
#             },
#             {
#                 "id": 1,
#                 "name": "MPV",
#                 "supercategory": "vechicle"
#             },
#             {
#                 "id": 2,
#                 "name": "SUV",
#                 "supercategory": "vechicle"
#             },
#             {
#                 "id": 3,
#                 "name": "Sedan/fastback",
#                 "supercategory": "vechicle"
#             },
#             {
#                 "id": 4,
#                 "name": "Minibus",
#                 "supercategory": "vechicle"
#             },
#             {
#                 "id": 5,
#                 "name": "Pickup",
#                 "supercategory": "vechicle"
#             },
#             {
#                 "id": 6,
#                 "name": "Estate",
#                 "supercategory": "vechicle"
#             },
#             {
#                 "id": 7,
#                 "name": "Sport",
#                 "supercategory": "vechicle"
#             }
#         ],
#         "images": [
#             {
#                 "id": 0,
#                 "file_name": "000007.jpg",
#                 "height": 305,
#                 "width": 247
#             },
#             {
#                 "id": 1,
#                 "file_name": "000001.jpg",
#                 "height": 557,
#                 "width": 756
#             },
#             {
#                 "id": 2,
#                 "file_name": "000012.jpg",
#                 "height": 295,
#                 "width": 327
#             }
#
#         ],
#         "annotations": [
#             {
#                 "id": 0,
#                 "image_id": 0,
#                 "category_id": 1,
#                 "bbox": [
#                     0,
#                     0,
#                     247,
#                     305
#                 ],
#                 "area": 75335,
#                 "iscrowd": 0
#             },
#             {
#                 "id": 0,
#                 "image_id": 1,
#                 "category_id": 2,
#                 "bbox": [
#                     0,
#                     0,
#                     756,
#                     557
#
#                 ],
#                 "area": 421092,
#                 "iscrowd": 0
#             },
#             {
#                 "id": 0,
#                 "image_id": 2,
#                 "category_id": 3,
#                 "bbox": [
#                     0,
#                     0,
#                     327,
#                     295
#
#                 ],
#                 "area": 96465,
#                 "iscrowd": 0
#             }
#         ]
#     }
#
#     json_string = json.dumps(json_data)
#     print(json_string)
#     with open('data.json', 'w') as f:
#         json.dump(json_data, f)

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
    root_folder = '/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1'
    root_images_folder = os.path.join(root_folder, 'images')
    vehicle_info_file_path = os.path.join(root_folder, 'train_test_split', 'vehicle_info.txt')

    with open(vehicle_info_file_path) as file:
        lines = [line.rstrip() for line in file]

    list_vehicle_types = []

    for line in lines:
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

        # with open('veriwild_annotations.json', 'w', encoding='utf-8') as f:
        #     json.dump(json_data, f, ensure_ascii=False, indent=4)


    print(list_vehicle_types)



if __name__ == "__main__":
    process_images()