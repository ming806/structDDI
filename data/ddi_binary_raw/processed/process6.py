import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
from util.myutil import *

drug_dic_len = 67  # vocab_size= 67+1   0 is for padding.
drug_dic = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
            "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
            "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
            "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
            "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
            "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
            "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
            "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, "X": 65, 'p': 66, 'k': 67}


def encoding_smile_sequence_from_list(list_path, max_len=100):
    """
    :param list_path: csv file with three column: smile1 smile2 label
    :param max_len: the sequence length
    :return: (smile1_array,smile2_array,label_array)
    """
    print("encoding file:", list_path)
    smiles_list = load_decompressed_pzip(list_path)
    number = len(smiles_list)
    smiles_array = np.zeros(shape=(number, max_len))
    for row, ss in tqdm(enumerate(smiles_list)):
        for col, cha in enumerate(list(ss)):
            if col == max_len:
                break
            smiles_array[row, col] = drug_dic[cha]
    print("finish encoding")
    return smiles_array


# 01 编码文本信息
# 02 编码每个分子图的邻接矩阵
# 03 编码每个分子图的特征矩阵
if __name__ == "__main__":
    # id1,smiles1,id2,smiles2,label
    # (83040, 5)
    biosnap_smiles_list = load_decompressed_pzip("../ddi_binary_raw_index_smiles_list.pzip")
    num = len(biosnap_smiles_list)
    print(len(biosnap_smiles_list))  # 9651
    file_path = "../ddi_binary_raw_index_smiles_list.pzip"
    # 01 将一维的smiles转换为二维的文本信息(graph_num,text_len)
    output = encoding_smile_sequence_from_list(file_path, max_len=100)
    store_compressed_pzip(output, "../ddi_binary_raw_index_text_array.pzip")
    # 02 将smiles转换为邻接矩阵与节点特征
    adj_list = []
    feat_list = []
    for idx, smiles in enumerate(biosnap_smiles_list):
        feat, adj = mol_features(smiles)
        adj_list.append(adj)
        feat_list.append(feat)
    print(len(adj_list))
    print(len(feat_list))

    node_sum = 0
    max_node_num = 0
    min_node_num = 1000000
    for tmp_adj in tqdm(adj_list):
        tmp_node_num = tmp_adj.shape[0]
        node_sum += tmp_node_num
        if tmp_node_num > max_node_num:
            max_node_num = tmp_node_num
        if tmp_node_num < min_node_num:
            min_node_num = tmp_node_num
    print("Graph number: %d ,maximum node number is %d, minimum node number is %d, mean node number is %d" %
          (len(adj_list), max_node_num, min_node_num, node_sum / len(adj_list)))

    for i in range(len(adj_list)):
        if np.isnan(adj_list[i].sum()):
            print("nan exist")
        if np.isnan(feat_list[i].sum()):
            print("nan exist")
    # store_pkl(adj_list, "../biosnap_index_adj.pkl")        # 邻接矩阵，因为每个图的矩阵大小不唯一，所以无法用h5进行存储
    # store_pkl(feat_list, "../biosnap_index_feat.pkl")      # 特征矩阵
    store_compressed_pzip(adj_list, "../ddi_binary_raw_index_adj.pzip")  # 邻接矩阵，因为每个图的矩阵大小不唯一，所以无法用h5进行存储
    store_compressed_pzip(feat_list, "../ddi_binary_raw_index_feat.pzip")  # 特征矩阵
