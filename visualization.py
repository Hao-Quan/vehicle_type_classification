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

# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(yolo_bboxes):
    xmin, ymin = yolo_bboxes[0]-yolo_bboxes[2]/2, yolo_bboxes[1]-yolo_bboxes[3]/2
    xmax, ymax = yolo_bboxes[0]+yolo_bboxes[2]/2, yolo_bboxes[1]+yolo_bboxes[3]/2
    return xmin, ymin, xmax, ymax


def plot_box(image, yolo_bboxes):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(yolo_bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # Denormalize the coordinates.
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)

        thickness = max(2, int(w / 275))

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )
    return image

def visualize_picture():

    bbox = [[0.5, 0.5, 1, 1]]
    im = cv2.imread("/media/hao/Seagate Basic/dataset/veri-wild/veri-wild1_debug/yolov8_resized_dataset_debug/train/images/000001.jpg")
    im_bb = plot_box(im, bbox)
    cv2.imshow("p1", im_bb)
    cv2.waitKey(0)


if __name__ == "__main__":
    # check_images_similar()
    visualize_picture()