from torch.utils.data import Dataset, DataLoader
from util.myutil import *
import pandas as pd
import time
from data.random_walker_new import make_graph_data_list_called


def random_divide(num, d_list):
    all_ids = [i for i in range(num)]
    test_num = int(num * d_list[2])  # 1770554　　 测试数据
    valid_num = int(num * d_list[1])
    train_num = num - test_num - valid_num
    print(train_num, valid_num, test_num)
    test_index = random.sample(all_ids, test_num)
    remain_ids = list(set(all_ids) - set(test_index))
    valid_index = random.sample(remain_ids, valid_num)
    train_index = list(set(remain_ids) - set(valid_index))
    assert num == test_num + train_num + valid_num
    assert num == len(train_index) + len(valid_index) + len(test_index)
    # print(len(set.intersection(set(valid_index), set(test_index))))
    # print(len(set.intersection(set(train_index), set(test_index))))
    # print(len(set.intersection(set(train_index), set(valid_index))))
    return train_index, valid_index, test_index


class NewDataset(Dataset):
    def __init__(self, dataset, walker_data, cur_index):
        super().__init__()
        self.used_index_list = cur_index
        self.interaction_indexed = np.array(dataset[2])[cur_index]  # 获取使用编号标注的交互记录  [index1,index2,label]
        self.walker_data = walker_data  # 获取walker的data

    def __getitem__(self, cur):  # when dataset[index],this function will be called.
        node1_index = int(self.interaction_indexed[cur, 0])  # 确保节点id是整型
        node2_index = int(self.interaction_indexed[cur, 1])  # 确保节点id是整型
        label = self.interaction_indexed[cur, 2]
        # [selected_node_list, final_feat_walk_array, final_graph_walk_array]
        struct1 = self.walker_data[node1_index][2].astype(int)  # (node_num,path_num)
        struct2 = self.walker_data[node2_index][2].astype(int)
        graph_feat1 = self.walker_data[node1_index][1]  # (node_num,path_num,path_length,feat_dim)
        graph_feat2 = self.walker_data[node2_index][1]
        return struct1, struct2, graph_feat1, graph_feat2, label

    def __len__(self):  # when use len(), this function will be called.
        return len(self.used_index_list)


def get_three_dataloader(data_path_list,
                         divide_list,
                         batch_size,
                         path_number,
                         path_length,
                         max_node_number,
                         is_random=True):
    dataset = [load_decompressed_pzip(file) for file in data_path_list]
    # for tmp in dataset:
    #     print(len(tmp))
    walker_data, structure_number = make_graph_data_list_called(dataset[0], dataset[1],
                                                                num_p=path_number,
                                                                path_l=path_length,
                                                                max_n=max_node_number)
    number = len(dataset[2])
    # 01 划分训练集/验证集/测试集
    if is_random:
        t_index, v_index, te_index = random_divide(number, divide_list)
    else:
        print("divide dataset orderly")
        train_number = int(number * divide_list[0])
        test_number = int(number * divide_list[2])
        valid_number = number - train_number - test_number
        t_index = [i for i in range(train_number)]
        v_index = [i for i in range(train_number, train_number + valid_number)]
        te_index = [i for i in range(train_number + valid_number, number)]
    index_lists = [t_index, v_index, te_index]

    print("train/valid/test: %d/%d/%d" % (len(t_index), len(v_index), len(te_index)))
    dataset_list = []
    for tmp_index in index_lists:
        dataset_list.append(NewDataset(dataset, walker_data, tmp_index))
    loader_list = []
    shuffle_list = [True, False, False]
    for idx, tmp in enumerate(dataset_list):
        loader_list.append(DataLoader(dataset=tmp, batch_size=batch_size, shuffle=shuffle_list[idx], num_workers=10,
                                      pin_memory=True))
    return loader_list[0], loader_list[1], loader_list[2], structure_number


# 建立deepDTA以及attentionDTA的数据集
if __name__ == "__main__":
    path = "./"
    suffix_list = [
        "_index_adj.pzip",  # <list>    # each element is array
        "_index_feat.pzip",
        "_interaction_cleaned_indexed.pzip"  # <list>    # each element is [5304, 4981, 1.0]
    ]
    data_id = 0
    dataset_name = ["ddi_binary_raw"]
    path_lists = []
    for name in dataset_name:
        path_lists.append([path + name + '/' + name + suffix for suffix in suffix_list])
    divide_list = [0.70, 0.10, 0.20]
    a, b, c, num = get_three_dataloader(data_path_list=path_lists[data_id],
                                        divide_list=divide_list,
                                        batch_size=32,
                                        path_number=20,
                                        path_length=3,
                                        max_node_number=10)
    num = 0
    for batch_idx, batch_data in enumerate(a):
        print(batch_idx)
        print(batch_data[0].shape)
        print(batch_data[1].shape)
        print(batch_data[2].shape)
        print(batch_data[3].shape)
        print(batch_data[4].shape)
        # print(batch_data[3].shape)
        # print(batch_data[4].shape)
        print(type(batch_data[0]))
        print(type(batch_data[1]))
        print(type(batch_data[2]))
        print(type(batch_data[3]))
        print(type(batch_data[4]))
        print(batch_data[0].dtype)
        print(batch_data[1].dtype)
        print(batch_data[2].dtype)
        print(batch_data[3].dtype)
        print(batch_data[4].dtype)
        # print(batch_data[1])
        # print(batch_data[2])
        # print(batch_data[3])
        # print(batch_data[4])
        if num == 0:
            break
