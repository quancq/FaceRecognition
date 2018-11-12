from utils import utils
from utils import project_utils
import face_recognition
import os
from model import baseline1


if __name__ == "__main__":
    # mids = ["m.01_0d4"]
    #
    # mid_name_map = project_utils.load_mid_name_map()
    # for mid in mids:
    #     print("MID : {} - Name (en) : {}".format(mid, mid_name_map[mid]["en"]))

    pass
    dir = "../Temp/Dataset/Original/Thúy Hằng - Thúy Hạnh"
    # dir = "/home/quanchu/Dataset/MyPictures/Nguyễn Thị Huyền"
    similarities = baseline1.get_sorted_similarity_images(dir)

    for fname, sim in similarities:
        print("File : {} - Sim : {:.4f}".format(fname, sim))
