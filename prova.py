import cv2
import numpy as np

# print(np.version.version)
# im = cv2.imread("/media/hao/My Passport/dataset/veri-wild/images/00001/000001.jpg")
# print(im.shape)
#
# im = cv2.imread("/media/hao/My Passport/dataset/veri-wild/images/00002/000007.jpg")
# print(im.shape)
#
# im = cv2.imread("/media/hao/My Passport/dataset/veri-wild/images/00003/000012.jpg")
# print(im.shape)

def check_images_similar():
    path1 = "/data/veri-wild/veri-wild1/images_part01/00091/000916.jpg"
    path2 = "/data/veri-wild/veri-wild1/images_part010/00091/000915.jpg"
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    res =  image1.shape == image2.shape and not (np.bitwise_xor(image1, image2).any())

    print(path1)
    print(path2)
    print(res)

if __name__ == "__main__":
    check_images_similar()