from utils_dir import utils
import os
import pandas as pd
from collections import defaultdict, Counter
import cv2
import argparse


def get_compact_mid_map(file_path="../Dataset/Top1M_MidList.Name.tsv", mid_list=[]):
    columns = ["MID", "Name"]
    df = utils.load_csv(file_path, delimiter="\t", header=None, names=columns)
    mid_map = {mid: 0 for mid in mid_list}

    # Convert dataframe to dictionary and compact dataframe
    compact_list = []
    mid_name_map = defaultdict(dict)
    for i, row in df.iterrows():
        mid, name_lang = row["MID"], row["Name"]
        lst = name_lang.split('@')
        name, lang = lst[0], lst[1]
        lang_name_map = mid_name_map[mid]
        lang_name_map.update({lang: name})

        if mid_map.get(mid) is not None:
            compact_list.append((mid, lang, name))
    compact_df = pd.DataFrame(compact_list, columns=["MID", "Language", "Name"])
    print("Convert dataframe to dictionary done")

    compact_map = {}
    for mid in mid_list:
        compact_map.update({mid: mid_name_map[mid]})
    # if len(mid_list) > 0:
    #     mid = mid_list[0]
    #     print("Sample map of {}: {}".format(mid, compact_map[mid]))
    #     print("Lang_Name_map of {} (size = {}): ".format(mid, len(compact_map[mid])))
    # print("Compact mid_lang_name_map size: ", len(compact_map))

    return compact_map, compact_df


def load_mid_name_map(file_path="..Temp/Dataset/Process/MID_Name.json"):
    map = utils.load_json(file_path)
    return map


def get_popular_element(elms):
    popular_element, num_occurrence = Counter(elms).most_common(1)[0]
    return popular_element, num_occurrence


def get_popular_element_batch(elm_matrix):
    popular_elms, num_occurrences = [], []
    for row in elm_matrix:
        popular_elm, num_occurrence = get_popular_element(row)
        popular_elms.append(popular_elm)
        num_occurrences.append(num_occurrence)

    return popular_elms, num_occurrences


def resize_images(src_img_paths, dst_img_paths=None, size=None):
    if size is not None:
        if dst_img_paths is None:
            dst_img_paths = src_img_paths
        elif len(dst_img_paths) != len(src_img_paths):
            print("Number destination image paths ({}) not equal with "
                  "number source image paths ({})".format(len(dst_img_paths), len(src_img_paths)))
            return 0

        num_resized_imgs = 0
        for i, (src_path, dst_path) in enumerate(zip(src_img_paths, dst_img_paths)):
            try:
                img = cv2.imread(src_path)
                img = cv2.resize(img, size)
                cv2.imwrite(dst_path, img)
                num_resized_imgs += 1
                print("{}/{} Resize image {} to new shape {} and save to {}".format(
                    i+1, len(src_img_paths), src_path, size, dst_path))
            except:
                print("Error when resize image ", src_path)

        print("Resize {} images successful".format(num_resized_imgs))


def resize_images_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("--src_image_dir", required=True)
    ap.add_argument("--size", help="New size (pixel) of images", default="160")

    args = vars(ap.parse_args())
    src_image_dir = args["src_image_dir"]
    size = args["size"]

    src_img_paths = utils.get_all_file_paths(src_image_dir)
    size = (int(size), int(size))
    resize_images(src_img_paths=src_img_paths, size=size)


def copy_subset_args():

    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dataset_dir", required=True)
    ap.add_argument("--dst_dataset_dir", required=True)
    ap.add_argument("--min_imgs_per_class", "-mipc", default="50")
    ap.add_argument("--max_classes", default="500")

    args = vars(ap.parse_args())
    src_dataset_dir = args["src_dataset_dir"]
    dst_dataset_dir = args["dst_dataset_dir"]
    min_imgs_per_class = int(args["min_imgs_per_class"])
    max_classes = int(args["max_classes"])

    num_classes = 0
    for class_name in utils.get_dir_names(src_dataset_dir):
        file_names = utils.get_file_names(os.path.join(src_dataset_dir, class_name))
        if len(file_names) >= min_imgs_per_class:
            num_classes += 1
            src_dst_paths = [(os.path.join(src_dataset_dir, class_name, src_name),
                              os.path.join(dst_dataset_dir, class_name, src_name))
                             for src_name in file_names]
            utils.copy_files(src_dst_paths)

            print("\nCopy {}/{} classes done".format(num_classes, max_classes))
            if num_classes >= max_classes:
                break


if __name__ == "__main__":
    # map_mid_name = load_mid_name_map()
    # print(len(map_mid_name))
    # src_img_paths = ["../Dataset/Test_Split/Train/m.01_0d4/26-FaceId-0.jpg"]
    # dst_img_paths = ["/home/quanchu/Pictures/temp1.jpg"]

    # resize_images_args()
    copy_subset_args()

    pass
