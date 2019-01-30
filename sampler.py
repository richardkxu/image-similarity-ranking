import os
import re
import numpy as np
import time


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|JPEG'):
    """
    list all images under directory with one of the extension
    this will recursively find all images, including those in subdirs
    :param directory:
    :param ext:
    :return:
    """
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if re.match(r'([\w]+\.(?:' + ext + '))', f)]


def get_negative_image(all_images, image_names):
    """
    get the path to one negative image, i.e., the image in all_images but
    not in image_names
    :param all_images:
    :param image_names:
    :return:
    """
    rand_idx = np.random.randint(len(all_images))
    while all_images[rand_idx] in image_names:
        rand_idx = np.random.randint(len(all_images))

    return all_images[rand_idx]


def get_positive_image(image_name, image_names):
    """
    get the path to one positive image, i.e., the image
    that belongs to the same class as image_name
    :param image_name:
    :param image_names:
    :return:
    """
    rand_idx = np.random.randint(len(image_names))
    while image_names[rand_idx] == image_name:
        rand_idx = np.random.randint(len(image_names))

    return image_names[rand_idx]


def triplet_sampler(directory_path):
    """
    sample a triplet for each image under directory_path
    a triplet = (query image, positive image, negative image)
    each run is random
    :param directory_path:
    :return:
    """
    classes = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    # a list of all image paths
    all_images = []
    for class_ in classes:
        all_images += (list_pictures(os.path.join(directory_path, class_)))

    triplets = []
    for class_ in classes:
        # a list of images that belong to the same class
        image_names = list_pictures(os.path.join(directory_path, class_))
        for image_name in image_names:
            query_image = image_name
            positive_image = get_positive_image(image_name, image_names)
            negative_image = get_negative_image(all_images, set(image_names))
            triplets.append([query_image, positive_image, negative_image])
    triplets = np.array(triplets)
    np.random.shuffle(triplets)

    return triplets


def emb_sampler(directory_path, ext='jpg|jpeg|bmp|png|ppm|JPEG'):

    classes = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

    emb_arr = []
    for class_ in classes:
        for root, _, files in os.walk(os.path.join(directory_path, class_)):
            for f in files:
                if re.match(r'([\w]+\.(?:' + ext + '))', f):
                    emb_arr.append([os.path.join(root, f), class_])

    emb_arr = np.array(emb_arr)

    return emb_arr


def val_sampler(root_path, f_path):
    """
    :param root_path:
    :param f_path:
    :return: return a list of val imgs, each row is [img_path, its label]
    """
    img_list = []
    with open(f_path, 'r') as rf:
        for line in rf.readlines():
            img_name= line.strip().split('\t')[0]
            img_path = os.path.join(root_path, img_name)
            label = line.strip().split('\t')[1]
            img_list.append([img_path, label])
    img_list = np.array(img_list)

    return img_list


if __name__ == '__main__':

    indir = './tiny-imagenet-200/train'
    time1 = time.time()
    triplet_arr = triplet_sampler(indir)
    time2 = time.time()
    print(triplet_arr.shape)
    print(triplet_arr)
    print(time2-time1)

    time3 = time.time()
    emb_arr = emb_sampler(indir)
    time4 = time.time()
    print(emb_arr.shape)
    print(emb_arr)
    print(time4-time3)

    val_path = './tiny-imagenet-200/val/images'
    val_f = './tiny-imagenet-200/val/val_annotations.txt'
    time5 = time.time()
    val_arr = val_sampler(val_path, val_f)
    time6 = time.time()
    print(val_arr.shape)
    print(val_arr)
    print(time6-time5)

