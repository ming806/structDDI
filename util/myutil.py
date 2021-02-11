import argparse
import glob
import os
import pickle
import random
import shutil
from math import sqrt
import numpy as np
import torch
import h5py
import bz2
import gzip
import _pickle as cPickle
from rdkit import Chem
import scipy.sparse as sp


# for large files, compress and decompress consume time
def store_compressed_pbz2(souce_object, path_file):
    with bz2.BZ2File(path_file, 'wb', compresslevel=9) as f:
        cPickle.dump(souce_object, f)


def load_decompressed_pbz2(path_file):
    db_file = bz2.BZ2File(path_file, 'rb')
    data = cPickle.load(db_file)
    db_file.close()
    return data


# for large files, compress and decompress consume time
def store_compressed_pzip(souce_object, path_file):
    with gzip.GzipFile(path_file, 'wb', compresslevel=6) as f:
        cPickle.dump(souce_object, f)


def load_decompressed_pzip(path_file):
    db_file = gzip.GzipFile(path_file, 'rb')
    data = cPickle.load(db_file)
    db_file.close()
    return data


# one-hot编码，没有占位符
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


# 带有占位符的one-hot编码
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def store_hdf5(object_list, path_file, group_list=None, level=3):
    if group_list is None:
        group_list = ['smile1', 'smile2', 'label']
    print(group_list)
    f = h5py.File(path_file, 'w')
    for idx, group in enumerate(group_list):
        f.create_dataset(name=group, data=object_list[idx], compression="gzip", compression_opts=level)
    f.close()


def load_hdf5(path_file, group_list=None):
    if group_list is None:
        group_list = ['smiles1', 'smiles2', 'label']
    f = h5py.File(path_file, 'r')
    value_list = []
    for key in group_list:
        value_list.append(np.array(f.get(key)[:]))
    f.close()
    return value_list


def store_pkl(souce_object, path_file):
    # wb: write file and use binary mode
    dbfile = open(path_file, 'wb')
    # source, destination
    pickle.dump(souce_object, dbfile)
    dbfile.close()


def load_pkl(path_file):
    # for reading also binary mode is important
    dbfile = open(path_file, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    # return np.array(adjacency) + np.eye(adjacency.shape[0])
    return np.array(adjacency)  # do not use self-loop


# use one-hot vector
# 包括分子类型,电荷量,
# 根据原子产生34个维度的编码。
# rdkit.Chem.rdchem module：module containing the core chemistry functionality of the RDKit
def atom_features(atom, use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    # 特征1：atom type:分子类型
    symbol = ['C', 'Co', 'P', 'K', 'Br', 'B', 'As', 'F', 'Ca', 'La', 'O', 'Au', 'Gd', 'Na', 'Se', 'N', 'Pt', 'S', 'Al',
              'Li', 'Cl', 'I', "other"]  # 23-dim
    # 特征2：degree of atom:分子的度    有多少化学键相连
    degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 11-dim
    # 特征3: hybridization type:杂化轨道类型
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']  # 6-dim
    #   atom.GetSymbol()：             原子类型       23个维度
    #   atom.GetDegree()：             原子的度编码    11个维度                     原子的度
    #   atom.GetFormalCharge()：       原子的formal charge: 1个维度                形式电荷
    #   atom.GetNumRadicalElectrons()：原子的number of radical electrons: 1个维度  自由基的数目
    #   atom.GetHybridization()：      hybridization type: 6个维度                杂化轨道类型：
    #    atom.GetIsAromatic()：        1个维度,                                   是否是芳香族
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + [atom.GetIsAromatic()]  # 43

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  #
    return results


def mol_features(smiles):
    # -----------------01 将SMILE字符串转换为分子表示--------------
    num_atom_feat = 46
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    # mol = Chem.AddHs(mol)
    # -----------------02 初始化分子图的节点特征矩阵--------------
    # 创建单个分子图的特征矩阵(分子数目, 分子的特征维度)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))

    # -----------------03 遍历单个分子, 初始化分子特征------------
    # 遍历单个分子获取每个分子
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    # no Z score
    # mean_value = np.mean(atom_feat, axis=0)
    # std_value = np.std(atom_feat, axis=0)
    # std_value[np.where(std_value == 0)] = 0.001
    # atom_feat = (atom_feat - mean_value) / std_value
    # -----------------04 获得分子的拓扑结构矩阵--------------
    adj_matrix = adjacent_matrix(mol)
    # 05 对分子特征矩阵进行row_normalized 同时将邻接矩阵变为无向的邻接矩阵
    # 感觉进行normalized的意义不是太大
    # atom_normalized_feat = preprocess_features(atom_feat)
    # assert the graph is undirected graph
    adj1 = sp.csr_matrix(adj_matrix)
    undirected_adj_matrix = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
    # 06 返回分子的normalized节点特征以及对称的拓扑结构矩阵
    return atom_feat, undirected_adj_matrix.todense()


def save_best_model(model, model_dir, best_epoch):
    # save parameters of trained model
    torch.save(model.state_dict(), model_dir + '{}.pkl'.format(best_epoch))
    files = glob.glob(model_dir + '*.pkl')
    # delete models saved before
    for file in files:
        tmp = file.split('/')[-1]  # windows:\\  linux: /
        tmp = tmp.split('.')[0]
        epoch_nb = int(tmp)
        if epoch_nb < best_epoch:
            os.remove(file)


# calculation sum of model parameters
def model_parameters_calculation(model):
    return sum([torch.numel(param) for param in model.parameters()])


def assert_dir_exist(x):
    if not os.path.exists(x):
        os.makedirs(x)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print("fix random seed !")


def value_to_hot(value_array):
    label_num = value_array.max() + 1
    num = value_array.shape[0]
    value_list = value_array.tolist()
    one_hot_array = np.zeros(shape=(num, label_num))
    for row, value in enumerate(value_list):
        one_hot_array[row, value] = 1
    return one_hot_array


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


# mean squared error
def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


# squared correlation coefficients with intercept
def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


# squared correlation coefficients without intercept
def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]  # ?????
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def get_file_name_from_dic(para_dic):
    keys_list = para_dic.keys()
    filename = ''
    for tmp_key in keys_list:
        filename += str(tmp_key) + ':' + str(para_dic[tmp_key]) + '|'
    filename += '.csv'
    return filename


# argv = parse_arguments(sys.argv[1:])
def parse_arguments_text_cci(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default='150')
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--dropout', type=float, default='0.5')
    parser.add_argument('--batch', type=int, default='32')
    parser.add_argument('--pooling_size', type=int, default='5')
    # parser.add_argument('-data_id', type=int, default='0')
    parser.add_argument('--channel_list', type=str, default='[16, 32, 64, 128]')
    parser.add_argument('--filter_size_smiles', type=str, default='[4, 6, 8]')
    parser.add_argument('--mlp_sizes', type=str, default='[256, 512, 256]')

    args = parser.parse_args(argv)
    return args


def str_list_to_int_list(str_list):
    str1 = str_list[1: -1].split(',')
    int_list = [int(i) for i in str1]
    return int_list


def change_para_dic_text_cci(parser, para_dic):
    para_dic['lr'] = parser.lr
    para_dic['epochs'] = parser.epochs
    para_dic['dropout'] = parser.dropout
    para_dic['batch'] = parser.batch
    para_dic['pooling_size'] = parser.pooling_size
    para_dic['channel_list'] = str_list_to_int_list(parser.channel_list)
    para_dic['filter_size_smiles'] = str_list_to_int_list(parser.filter_size_smiles)
    para_dic['mlp_sizes'] = str_list_to_int_list(parser.mlp_sizes)


def write_to_txt(file_path, content_str):
    with open(file_path, 'a+') as f:
        f.write(content_str + '\n')


def str_list_of_list_to_int_list(str_lists):
    strs = str_lists[1: -1].split('_')
    final_lists = [str_list_to_int_list(tmp) for tmp in strs]
    return final_lists


if __name__ == "__main__":
    print("test")
    # para_dic = {
    #     'smile_embedding_dim': 32,
    #     'dropout': 0.1,
    #     'channel_list': [32, 64, 128, 256],
    #     'filter_size_smiles': [4, 6, 8],
    #     'batch': 32,
    #     'lr': 1e-3,  # learning rate
    #     'seed': 20,  # random seed
    #     'epochs': 150,
    #     'CUDA': True,
    #     'stop_counter': 200,
    #     'is_train': True,
    #     'is_test': True,
    #     'class_number': 2,
    # }
    # print(para_dic)
    # print(os.altsep)
    # 在Windows系统下的分隔符是：\ (反斜杠)。
    # 在Linux系统下的分隔符是： / （斜杠）。
    # model_dir = '../savedModel/savedTextCCI/' + 'cci900' + '/'
    # best_epoch = 10
    # files = glob.glob(model_dir + '*.pkl')
    # # delete models saved before
    # for file in files:
    #     print(file)
    #     tmp = file.split('\\')[-1]
    #     print(tmp)
    #     tmp = tmp.split('.')[0]
    #     epoch_nb = int(tmp)
    #     if epoch_nb < best_epoch:
    #         os.remove(file)
    # print(torch.cuda.current_device())
    # print(torch.cuda.device_count())
    # print(torch.cuda.get_device_name(0))
    # print(torch.cuda.is_available())
    # with open('./data.txt', 'r') as f:
    #     data = f.read()
    #     print('context: {}'.format(data))
    # str_list_of_list_to_int_list('[[3, 1]_[5, 1]]')
    # store_hdf5(None, 'test.hdf5', group_list=None)
    # object_list = load_hdf5('test.hdf5')
    # for tmp in object_list:
    #     print(tmp.shape)
