import copy
from scipy.sparse import lil_matrix
import networkx as nx
import numpy as np
import random
from util.myutil import load_decompressed_pzip
from tqdm import tqdm


def generate_random_walk(begin_node, path_length, drug_graph):
    """
    :param begin_node: node id of start node
    :param path_length: sequence length
    :return: node id sequence
    """
    walk = [begin_node]
    while len(walk) < path_length:
        cur = walk[-1]
        cur_neighbors = [n for n in drug_graph.neighbors(cur)]  # 获取当前节点的邻居节点
        if len(cur_neighbors) == 0:  # for some SMILE graph with only few nodes and isolated nodes
            next_node = cur
        else:
            next_node = random.choice(cur_neighbors)
        walk.append(next_node)
    return walk


def generate_random_walk_new(begin_node, p_length, graph, p_num):
    """
    :param p_num:
    :param graph:
    :param p_length:
    :param begin_node: node id of start node
    :return: node id sequence
    """
    final_lists = []
    for num in range(p_num):
        walk = [begin_node]
        while len(walk) < p_length:
            cur = walk[-1]
            cur_neighbors = [n for n in graph.neighbors(cur)]  # 获取当前节点的邻居节点
            if len(cur_neighbors) == 0:  # for some SMILE graph with only few nodes and isolated nodes
                next_node = cur
            else:
                next_node = random.choice(cur_neighbors)
            walk.append(next_node)
        final_lists.append(walk)
    return final_lists


def generate_anonym_walks(length):
    """
    recursive function to generate all anonymous sequence of this length.
    :param length: length of anonymous walker
    :return:  list which includes many lists,each is anonymous sequence
    """
    anonymous_walks = []

    def generate_anonymous_walk(totlen, pre):  # inner function definition
        if len(pre) == totlen:
            anonymous_walks.append(pre)
            return
        else:
            candidate = max(pre) + 1
            for i in range(1, candidate + 1):
                if i != pre[-1]:
                    npre = copy.deepcopy(pre)
                    npre.append(i)
                    generate_anonymous_walk(totlen, npre)

    generate_anonymous_walk(length, [1])
    return anonymous_walks


# 按道理图中不应该出现孤立的节点，但是SMILE字符串转换后得到的确实有孤立的节点
# 且对孤立节点进行随机游走的得到的就是全是１的匿名序列
def generate_walk2num_dict(length):
    anonym_walks = generate_anonym_walks(length)
    anonym_dict = dict()
    tmp_list = [1 for i in range(length)]  # 孤立节点的随机游走序列都是1,其映射的类型为0
    isolated_pattern = "".join([str(x) for x in tmp_list])
    anonym_dict[isolated_pattern] = 0  # for isolated SMILE nodes and padding
    curid = 1
    for walk in anonym_walks:
        swalk = "".join([str(x) for x in walk])  # int list to string
        anonym_dict[swalk] = curid
        curid += 1
    return anonym_dict


# 虽有游走序列转换为匿名随机游走序列
def to_anonym_walk(walk):
    """

    :param walk: node id sequcence list
    :return: annoym_walk: int sequence list
    """
    num_app = 0
    apped = dict()  # save the node id first appear
    anonym = []
    for node in walk:
        if node not in apped:
            num_app += 1
            apped[node] = num_app
        anonym.append(apped[node])
    return anonym


# 紧密中心性的计算　（v-1）/（到其他节点距离的总和）　＝　某个节点到其他节点距离的平均值的倒数
def closeness_centrality_calculation(graph, max_num=10):
    """
    :param graph: networkx graph
    :param max_num: the number to be selected which has the biggest closeness centrality calculation
    :return: node id list
    """
    nodes_num = len(graph.nodes())
    centrality_list = []
    for src_id in range(nodes_num):
        shortest_dic = nx.shortest_path_length(graph, source=src_id)
        distance_sum = 0
        for target_id in range(nodes_num):
            if target_id in shortest_dic:  # 转换后的分子图中可能存在孤立的子图,因此需要判断一下,原因待探究
                distance_sum += shortest_dic[target_id]
        if distance_sum == 0:  # the isolated nodes.
            centrality = 0
        else:
            centrality = (nodes_num - 1) / distance_sum
        centrality_list.append(centrality)
    tmp = zip(range(nodes_num), centrality_list)
    sorted_tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    res_id_list = []
    # when the whole node number is less than max_num, do random sample.
    # [(1, 1.0), (0, 0.5714285714285714), (2, 0.5714285714285714), (3, 0.5714285714285714), (4, 0.5714285714285714)]
    # print(sorted_tmp)  # [tuple1,tuple2,...]  tuple1:(node_id,centrality_calc)
    current_node_num = len(sorted_tmp)
    used_node_num = min(current_node_num, max_num)  # 实际选择出来的节点
    selected_node_num = []
    for t in range(used_node_num):
        res_id_list.append(sorted_tmp[t][0])
        selected_node_num.append(sorted_tmp[t][0])
    remain_node_num = max_num - used_node_num
    # 当节点数目不够时,是否进行重复选择
    # if remain_node_num > 0:
    #     repeat_node_list = np.random.choice(selected_node_num, size=remain_node_num, replace=True)
    #     res_id_list.extend(repeat_node_list)
    # assert len(res_id_list) == max_num
    return res_id_list


# the most important function
def generate_graph_data(adj_numpy_matrix, feat_graph, str_to_index_dic, num_paths=50, path_length=6, max_num=10):
    """
    :param str_to_index_dic: 匿名随机游走序列映射字典.  key:匿名随机游走字符串 value:匿名随机游走类型编号,也就是所谓的结构模编号
    :param adj_numpy_matrix: (node_num,node_num) 传入的特征矩阵在做预处理的时候就是对称矩阵,没有self-loop
    :param feat_graph: (node_num,feat_dim)
    :param num_paths: 50
    :param path_length: 6
    :param max_num: 10
    :return: numpy
            (10,)             selected nodes list
            (10, 52)          graph overall features  (removed)
            (10, 50)          graph structure pattern representation
            (10, 50, 6, 38)   graph walk features representation (removed)
    """
    # 01 使用networksx将numpy的邻接矩阵建立图
    mm = lil_matrix(adj_numpy_matrix)
    real_node_num = mm.shape[0]
    # print("the node number of real graph %d" % real_node_num)
    rows = mm.rows
    adjacency_dic = dict(zip([i for i in range(rows.shape[0])], rows))
    # print(adjacency_dic)
    edge_list = []
    for one_edge in adjacency_dic.keys():
        for another_edge in adjacency_dic[one_edge]:
            edge_list.append((one_edge, another_edge))
    # print(edge_list)
    drug_graph = nx.Graph(edge_list)
    # the smile graph may have some isolated nodes,this step help adding isolated nodes
    for node_id in range(real_node_num):
        drug_graph.add_node(node_id)

    # 02 获取需要随机游走的节点编号
    selected_node_list = closeness_centrality_calculation(graph=drug_graph, max_num=max_num)
    # print("The selected nodes ids for random walk:" + str(selected_node_list))

    # 03 获取节点id到随机游走序列的映射
    walk_dic = {}  # key:node_id, value: walk list
    for selected_id in selected_node_list:
        tmp_walk_lists = generate_random_walk_new(begin_node=selected_id, p_length=path_length, graph=drug_graph,
                                                  p_num=num_paths)
        walk_dic[selected_id] = tmp_walk_lists  # (path_num,path_length)
    # print("random walker for each selected nodes" + str(walk_dic))

    # 04 将每个节点获得的匿名随机游走序列转化得到(node_num,dim)维度的特征表示
    id_walk_feat_dic = {}
    node_feat_dim = feat_graph.shape[1]
    for selected_id in walk_dic.keys():  # 遍历选择出来的节点
        passed_node_lists = walk_dic[selected_id]
        feat_array = np.zeros(shape=(num_paths, path_length, node_feat_dim))  # 每个节点的特征维度是(路径数,路径长度,节点的维度)
        for path_idx, tmp_list in enumerate(passed_node_lists):  # 遍历该节点的每一个路径
            feat_array[path_idx, :, :] = feat_graph[tmp_list]    # 使用list作为作为索引返回新的array,要求list的范围不超过array维度
        id_walk_feat_dic[selected_id] = feat_array  # (path_number,path_length,feat_dim)
    # print("Graph walker feature:" + str(id_walk_feat_dic))

    # 05 将随机游走序列转换为匿名随机游走序列
    anonymous_walk_dic = {}
    for key, value in walk_dic.items():
        tmp_list = []
        for id_list in walk_dic[key]:
            tmp_list.append(to_anonym_walk(id_list))
        anonymous_walk_dic[key] = tmp_list
    # 匿名随机游走序列都是1代表这是一个孤立的节点
    # print("anonymous walker for each selected nodes" + str(anonymous_walk_dic))

    # 0表示孤立的节点, 1~n分别是对一种匿名随机游走序列进行映射.
    # 06 将随机游走序列映射为结构模式索引(节点的特征该如何编码)
    # 对匿名随机游走表示的结果进行embedding嵌入,从而让各个分子之间共享相同的嵌入表示
    # str_to_index_dic = generate_walk2num_dict(path_length)
    type_walk_dic = {}  # key: selected node id  value: walker type list
    for key, value in anonymous_walk_dic.items():
        tmp_type_list = []
        for id_list in anonymous_walk_dic[key]:
            str_walk = "".join(str(x) for x in id_list)
            tmp_type_list.append(str_to_index_dic[str_walk])
        type_walk_dic[key] = tmp_type_list
    # print("Graph structure pattern:" + str(type_walk_dic))

    # 07 得到该图的   结构模式向量+随机游走序列的特征表示
    # 比较重要的数据结构:
    # type_walk_dic:      key是目标节点id,value是该目标节点周围的结构模型list,长度为path_number
    # id_walk_feat_dic:   key是目标节点,value是目标节点的随机游走序列特征表示,长度为(path_number,path_length,feat_dim)
    # selected_node_list: 该分子图的目标节点id列表
    final_graph_walk_array = np.zeros(shape=(max_num, num_paths))
    final_feat_walk_array = np.zeros(shape=(max_num, num_paths, path_length, node_feat_dim))
    for idx, tmp_id in enumerate(selected_node_list):
        final_graph_walk_array[idx, :] = type_walk_dic[tmp_id]
        final_feat_walk_array[idx, :, :, :] = id_walk_feat_dic[tmp_id]

    return [selected_node_list, final_feat_walk_array, final_graph_walk_array]


def make_graph_data_list(adj_p, feat_p, num_p=50, path_l=6, max_n=10):
    print("Path number %d,Path length %d,Maximum Node number %d" % (num_p, path_l, max_n))
    adj_matrix_list = load_decompressed_pzip(adj_p)  # list  <class 'numpy.ndarray'> (node_num,node_num)
    feat_matrix_list = load_decompressed_pzip(feat_p)  # list  <class 'numpy.matrix'> (node_num,feat_dim)
    graph_number = len(feat_matrix_list)
    str_to_index_dic = generate_walk2num_dict(path_l)  # structure pattern map dictionary
    extraction_list = []
    for i in tqdm(range(graph_number)):
        tmp_data = generate_graph_data(adj_matrix_list[i], feat_matrix_list[i], str_to_index_dic=str_to_index_dic,
                                       num_paths=num_p, path_length=path_l, max_num=max_n)
        extraction_list.append(tmp_data)

    return extraction_list


def make_graph_data_list_called(adj_matrix_list, feat_matrix_list, num_p=50, path_l=6, max_n=10):
    # adj_matrix_list = load_decompressed_pzip(adj_p)  # list  <class 'numpy.ndarray'> (node_num,node_num)
    # feat_matrix_list = load_decompressed_pzip(feat_p)  # list  <class 'numpy.matrix'> (node_num,feat_dim)
    print("Path number %d,Path length %d,Maximum Node number %d" % (num_p, path_l, max_n))
    graph_number = len(feat_matrix_list)
    str_to_index_dic = generate_walk2num_dict(path_l)  # structure pattern map dictionary
    extraction_list = []
    for i in tqdm(range(graph_number)):
        tmp_data = generate_graph_data(adj_matrix_list[i], feat_matrix_list[i], str_to_index_dic=str_to_index_dic,
                                       num_paths=num_p, path_length=path_l, max_num=max_n)
        extraction_list.append(tmp_data)

    return extraction_list, len(str_to_index_dic)   # number of structure number


# 子图中选择的节点,子图中总体的子结构分布,子图中选择的节点匿名随机游走表示,子图中随机游走序列的特征表示
# (10,)
# (10, 52)
# (10, 50)
# (10, 50, 6, 34)
'''
path_length                     2       3    4    5     6     7      8       9        10 

structure pattern number        1       2    5    15    53    203    877    4140     21147
                                       17s
考虑到如果该节点是孤立的节点,那么这个也算一种结构模式,因此embedding的维度要加上1,这种结构模式就是1,1,1,1,1
'''

if __name__ == "__main__":
    # 这里在实现的时候绕了弯路，应该先选中心节点，在进行匿名随机游走
    # ========================00 相关超参数=====================================
    num_paths = 50  # 路径的个数
    path_length = 10  # 随机游走路径长度
    max_num = 3  # 子图中根据closeness centrality 挑选出节点数目
    adj_path = 'biosnap_raw/biosnap_raw_index_adj.pzip'
    feat_path = 'biosnap_raw/biosnap_raw_index_feat.pzip'
    # adj_path = 'ddi_binary_raw/ddi_binary_raw_index_adj.pzip'
    # feat_path = 'ddi_binary_raw/ddi_binary_raw_index_feat.pzip'
    make_graph_data_list(adj_path, feat_path, num_p=num_paths, path_l=path_length, max_n=max_num)
    structure_num = generate_anonym_walks(path_length)
    print("number of structure is %d" % len(structure_num))
