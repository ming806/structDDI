import pandas as pd
import numpy as np
import h5py
from util import *

if __name__ == "__main__":
    # id1,smiles1,id2,smiles2,label
    # (83040, 5)
    df_interact = pd.read_csv("../biosnap_interaction_raw.csv", usecols=['smiles1', 'smiles2', 'label'])
    print(df_interact.shape)
    # map_dic = load_pkl("../biosnap_smiles_nodeId_dic.pkl")
    map_dic = load_decompressed_pzip("../biosnap_raw_smiles_nodeId_dic.pzip")
    print(len(map_dic))   # 9651

    df_array = df_interact.values
    number = df_array.shape[0]
    edge_list = []
    for i in range(number):
        s1 = df_array[i, 0]
        s2 = df_array[i, 1]
        label = df_array[i, 2]
        edge_list.append([map_dic[s1], map_dic[s2], label])
    edge_df = pd.DataFrame(edge_list, columns=["smiles1", "smiles2", "label"])
    edge_df.to_csv("../biosnap_raw_interaction_cleaned_indexed.csv", index=False)
    print(edge_df.shape)
    # h5_object = [np.array(edge_list).astype(int)]
    # store_hdf5(h5_object, "../biosnap_interaction_cleaned_indexed.h5",
    #            group_list=["biosnap_interaction_cleaned_indexed"])       # 生成用节点idi表示的数据集
    store_compressed_pzip(edge_list, "../biosnap_raw_interaction_cleaned_indexed.pzip")
