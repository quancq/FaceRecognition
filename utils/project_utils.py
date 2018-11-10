import utils
import os
import pandas as pd
from collections import defaultdict


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


def load_mid_name_map(file_path="../Dataset/MID_Name.json"):
    map = utils.load_json(file_path)
    return map


if __name__ == "__main__":
    # map_mid_name = load_mid_name_map()
    # print(len(map_mid_name))
    pass
