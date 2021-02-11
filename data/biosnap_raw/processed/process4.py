import pandas as pd
import numpy as np
import h5py
from util import *


def build_smile_id_dic(id_smile_df):
    smiles_id_dic = {}
    df_array = id_smile_df.values
    number = df_array.shape[0]
    for i in range(number):
        smiles_id_dic[df_array[i, 1]] = df_array[i, 0]
    return smiles_id_dic


if __name__ == "__main__":
    # id1,smiles1,id2,smiles2,label
    # (83040, 5)
    node_id_smile = pd.read_csv("../biosnap_raw_index.csv")
    print(node_id_smile.shape)
    map_dic = build_smile_id_dic(node_id_smile)
    # store_pkl(map_dic, "../biosnap_smiles_nodeId_dic.pkl")    # 生成smile字符串到节点id的映射的映射
    store_compressed_pzip(map_dic, "../biosnap_raw_smiles_nodeId_dic.pzip")
