import os
import json
import shutil
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import scipy.io
import pandas as pd
import splitfolders

def compcars_data_label_preparation():
    # local
    # root_path = "/media/hao/Seagate Basic/dataset/compcars/compcars_debug"
    # root_torchvision_path = "/media/hao/Seagate Basic/dataset/compcars/compcars_torchvision_debug"

    # remote
    root_path = "/data/compcars/compcars"
    root_torchvision_path = "/data/compcars/compcars_torchvision"

    original_image_folder_path = os.path.join(root_path, 'data/image')
    attribute_file_path = os.path.join(root_path, "data/misc/attributes.txt")


    torchvision_images_path = os.path.join(root_torchvision_path, "data/image")

    car_types_orig_list = ['MPV',
                           'SUV',
                           'sedan',
                           'hatchback',
                           'minibus',
                           'fastback',
                           'estate',
                           'pickup',
                           'hardtop convertible',
                           'sports',
                           'crossover',
                           'convertible'
                           ]
    str_mpv = 'MPV'
    str_suv = 'SUV'
    str_sedan_fastback = 'sedan_fastback'
    str_hatchback = 'hatchback'
    str_minibus = 'minibus'
    str_estate = 'estate'
    str_pickup = 'pickup'
    str_sport = 'sport'

    car_types_mapped_list = [str_mpv,
                             str_suv,
                             str_sedan_fastback,
                             str_hatchback,
                             str_minibus,
                             str_estate,
                             str_pickup,
                             str_sport]

    for car_type in car_types_mapped_list:
        if not os.path.exists(os.path.join(torchvision_images_path, car_type)):
            os.makedirs(os.path.join(torchvision_images_path, car_type))

    # with open(attribute_file_path) as file:
    #     lines = [line.rstrip() for line in file]

    attribute_df = pd.read_csv(attribute_file_path, sep=" ")

    for subdir, dirs, files in os.walk(original_image_folder_path):
        for file in files:
            current_modelid = subdir.split("/")[-2]
            selected_row_cartype = attribute_df.loc[attribute_df['model_id'] == int(current_modelid)]['type'].values[0]
            if selected_row_cartype != 0:
                source_filepath = os.path.join(subdir, file)
                print(source_filepath)

                img = cv2.imread(source_filepath)
                img = cv2.resize(img, (640, 640))
                # cv2.imwrite(os.path.join(target_filepath, file), img)

                if selected_row_cartype == 1:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_mpv))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_mpv, file), img)
                elif selected_row_cartype == 2:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_suv))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_suv, file), img)
                elif selected_row_cartype == 3:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_sedan_fastback))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_sedan_fastback, file), img)
                elif selected_row_cartype == 4:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_hatchback))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_hatchback, file), img)
                elif selected_row_cartype == 5:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_minibus))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_minibus, file), img)
                elif selected_row_cartype == 6:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_sedan_fastback))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_sedan_fastback, file), img)
                elif selected_row_cartype == 7:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_estate))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_estate, file), img)
                elif selected_row_cartype == 8:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_pickup))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_pickup, file), img)
                elif selected_row_cartype == 9:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_sedan_fastback))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_sedan_fastback, file), img)
                elif selected_row_cartype == 10:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_sport))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_sport, file), img)
                elif selected_row_cartype == 11:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_suv))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_suv, file), img)
                elif selected_row_cartype == 12:
                    # shutil.copy(os.path.join(subdir, file), os.path.join(torchvision_images_path, str_sedan_fastback))
                    cv2.imwrite(os.path.join(torchvision_images_path, str_sedan_fastback, file), img)

    a = 1

def dataset_train_valid_test_split():
    # local
    # root_path = "/media/hao/Seagate Basic/dataset/compcars/compcars_torchvision_debug/data/image"
    # output_path = "/media/hao/Seagate Basic/dataset/compcars/compcars_torchvision_debug/data/image_splitted"

    root_path = "/data/compcars/compcars_torchvision/data/image"
    output_path = "/data/compcars/compcars_torchvision/data/image_splitted"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    splitfolders.ratio(root_path, output=output_path, seed=1337, ratio=(.8, 0.1, 0.1))

    # train_ratio = 0.7
    # valid_ratio = 0.2
    # test_ratio = 0.1
    #
    # # root_path = "/media/hao/Seagate Basic/dataset/compcars/compcars_debug"
    # root_path = "/media/hao/Seagate Basic/dataset/compcars/compcars_torchvision_debug/data/image"
    # metrics = ["train", "valid", "test"]
    #
    # str_mpv = 'MPV'
    # str_suv = 'SUV'
    # str_sedan_fastback = 'sedan_fastback'
    # str_hatchback = 'hatchback'
    # str_minibus = 'minibus'
    # str_estate = 'estate'
    # str_pickup = 'pickup'
    # str_sport = 'sport'
    #
    # car_types_mapped_list = [str_mpv,
    #                          str_suv,
    #                          str_sedan_fastback,
    #                          str_hatchback,
    #                          str_minibus,
    #                          str_estate,
    #                          str_pickup,
    #                          str_sport]
    #
    # for metric in metrics:
    #     for car_type in car_types_mapped_list:
    #         if not os.path.exists(os.path.join(root_path, metric, car_type)):
    #             os.makedirs(os.path.join(root_path, metric, car_type))
    #
    # root_path
    # for str_folder in [str_mpv, str_suv, str_sedan_fastback, str_hatchback, str_minibus, str_estate, str_pickup, str_sport]:
    #     folder_num_files = len([name for name in os.listdir(os.path.join(root_path, str_folder)) if os.path.isfile(name)])
    #     folder_train_num = folder_num_files * train_ratio
    #     folder_valid_num = folder_num_files * valid_ratio
    #     folder_test_num = folder_num_files * test_ratio
    #     # for i in range(0, folder_train_num):






if __name__ == "__main__":
    # compcars_data_label_preparation()
    dataset_train_valid_test_split()
