import pandas as pd
import numpy as np
from util import *


def file_statistics(data_array):
    s_set = set()  # smile字符串的集合
    s_list = []  # 用于存储拼接的SMILE字符串的list
    s_list_dic = {}  # 用于统计拼接的smile字符串出现的次数
    num = data_array.shape[0]
    for r in range(num):
        s1 = data_array[r, 1]
        s2 = data_array[r, 3]
        s_set.add(s1)
        s_set.add(s2)
        s_concat = s1 + '|' + s2
        s_list.append(s_concat)
        if s_concat not in s_list_dic:
            s_list_dic[s_concat] = [r]
        else:
            s_list_dic[s_concat].append(r)
    unique_interaction = len(set(s_list))
    repeat_list = []
    # 用于存储重复数据
    if unique_interaction < num:
        print("the dataset has multiple records orign/real = %d/%d" % (num, unique_interaction))
        for key, value in s_list_dic.items():
            if len(s_list_dic[key]) > 1:
                # print(smiles_list_dic[key])
                repeat_list.append(s_list_dic[key])
    else:
        print("the dataset has no replicate records!")
    print("Unique smile string is %d" % len(s_set))
    return s_set, s_list, s_list_dic, repeat_list


def generate_index_csv(smile_set):
    print(len(smile_set))  # 9651
    unique_smile_list = list(smile_set)
    sorted_unique_smile_list = sorted(unique_smile_list, key=lambda x: len(x))
    length_sum = 0
    more_than_100 = 0
    for i in range(len(sorted_unique_smile_list)):
        length_sum += len(sorted_unique_smile_list[i])
        if len(sorted_unique_smile_list[i]) >= 100:
            more_than_100 += 1
    avg = length_sum/len(sorted_unique_smile_list)
    print("average length %f, number of smile length >= 100: %d" % (avg, more_than_100))

    index_list = []
    for idx, smiles in enumerate(sorted_unique_smile_list):
        index_list.append([idx, smiles])
    return index_list


if __name__ == "__main__":
    df = pd.read_csv("../ddi_binary_interaction_raw.csv")  # (83040, 5)
    s_set, _, _, _ = file_statistics(df.values)
    i_list = generate_index_csv(s_set)
    out_df = pd.DataFrame(i_list)
    out_df.columns = ["node_id", "smiles"]
    out_df.to_csv("../ddi_binary_raw_index.csv", index=False)
    print(out_df.shape)    # (9651, 2)  按照smile字符串从短到长进行排列，然后生成该数据唯一的节点id。
    sorted_smile_list = [tmp[1] for tmp in i_list]
    print(len(sorted_smile_list))
    store_compressed_pzip(sorted_smile_list, "../ddi_binary_raw_index_smiles_list.pzip")