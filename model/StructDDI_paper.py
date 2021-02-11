import torch
import torch.nn as nn
import torch.nn.functional as F
from util.myutil import model_parameters_calculation


class StructConvolution(nn.Module):
    def __init__(self, channels_list, filter_sizes, pool_method='max'):
        """
        :param channels_list: number of filters list
        :param filter_sizes:  filter length list
        """
        super().__init__()
        # (N,C,H,W) = (batch,embed_dim,path_num,node_num)
        self.layers = nn.ModuleList()
        for i in range(len(filter_sizes)):
            self.layers.append(nn.Conv2d(in_channels=channels_list[i],
                                         out_channels=channels_list[i + 1],
                                         kernel_size=filter_sizes[i]))
        if pool_method == 'max':
            self.pooling_layer = nn.AdaptiveMaxPool2d((1, None))
        elif pool_method == 'avg':
            self.pooling_layer = nn.AdaptiveAvgPool2d((1, None))
        self.init_params()

    def init_params(self):
        for m in self.layers:
            nn.init.xavier_uniform_(m.weight.data)  # glorot_uniform initialization,for Relu function
            nn.init.constant_(m.bias.data, 0.1)  # set bias

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = F.relu(self.layers[i](out))  # (batch,dim,path_num,node_num)
        out = self.pooling_layer(out)
        out = out.squeeze()  # (batch,dim,node_num)
        return out


class GraphFeatConvolution(nn.Module):
    def __init__(self, channels_list, filter_sizes, pool_method='max'):
        """
        :param channels_list: number of filters list
        :param filter_sizes:  filter length list
        """
        super().__init__()
        # (N,C,D,H,W) = (batch,embed_dim,path_num,path_length, node_num)
        self.layers = nn.ModuleList()
        for i in range(len(filter_sizes)):
            self.layers.append(nn.Conv3d(in_channels=channels_list[i],
                                         out_channels=channels_list[i + 1],
                                         kernel_size=filter_sizes[i]))
        if pool_method == 'max':
            self.pooling_layer = nn.AdaptiveMaxPool3d((1, 1, None))
        elif pool_method == 'avg':
            self.pooling_layer = nn.AdaptiveAvgPool3d((1, 1, None))
        self.init_params()

    def init_params(self):
        for m in self.layers:
            nn.init.xavier_uniform_(m.weight.data)  # glorot_uniform initialization,for Relu function
            nn.init.constant_(m.bias.data, 0.1)  # set bias

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = F.relu(self.layers[i](out))  # (batch,dim,path_num,path_length,node_num)
        out = self.pooling_layer(out)
        out = out.squeeze()  # (batch,dim,node_num)
        return out


# class FeatConvolution(nn.Module):
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x)  # this is different from DeepDTA which use relu.


class StructDDI(nn.Module):
    def __init__(self, para_dic):
        super(StructDDI, self).__init__()
        structure_embedding_number = para_dic['structure_embedding_number']
        struct_channel_list = para_dic['struct_channel_list']
        struct_filter_sizes = para_dic['struct_filter_sizes']
        feat_channel_list = para_dic['feat_channel_list']
        feat_filter_sizes = para_dic['feat_filter_sizes']
        conv_output_channel = para_dic['conv_output_channel']
        dropout = para_dic['dropout']
        class_number = para_dic['class_number']
        mlp_sizes = para_dic['mlp_sizes']
        task_type = para_dic['task_type']
        self.use_local_common = para_dic['use_local_common']
        # 01 input layer
        if self.use_local_common:
            self.structure_embedding = nn.Embedding(num_embeddings=structure_embedding_number,
                                                    embedding_dim=struct_channel_list[0])
            nn.init.uniform_(self.structure_embedding.weight, 0, 1)
            self.structure_conv = StructConvolution(struct_channel_list, struct_filter_sizes)

        # 02 layer for information fusion in graph structure features space
        self.feat_conv = GraphFeatConvolution(feat_channel_list, feat_filter_sizes)

        if self.use_local_common:
            # linear transformation
            self.transform_layer = nn.Linear(in_features=struct_channel_list[-1] + feat_channel_list[-1],
                                             out_features=conv_output_channel)
        else:
            self.transform_layer = nn.Linear(in_features=feat_channel_list[-1],
                                             out_features=conv_output_channel)

        nn.init.xavier_uniform_(self.transform_layer.weight)
        nn.init.constant_(self.transform_layer.bias, 0)
        self.Global_Max_pool = nn.AdaptiveMaxPool1d(1)
        # 04 combined layer and predication
        in_dim = conv_output_channel
        # ========03 combined layer and predication======================
        self.fc = nn.Sequential(
            Linear(in_features=in_dim, out_features=mlp_sizes[0]),
            nn.Dropout(dropout),
            Linear(in_features=mlp_sizes[0], out_features=mlp_sizes[1]),
            nn.Dropout(dropout),
            Linear(in_features=mlp_sizes[1], out_features=mlp_sizes[2]),
            nn.Dropout(dropout),
        )

        if task_type == 'classification':
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=mlp_sizes[-1], out_features=class_number),
                nn.LogSoftmax(dim=1)
            )
        elif task_type == 'regression':
            self.output_layer = nn.Sequential(
                nn.Linear(in_features=mlp_sizes[1], out_features=1),
            )
        nn.init.normal_(self.output_layer[0].weight)

    def forward(self, graph_struct1, graph_struct2, graph_feat1, graph_feat2):
        feat1 = graph_feat1.permute(0, 4, 2, 3, 1)  # (batch,feat_dim,path_num,len,node_num)
        feat2 = graph_feat2.permute(0, 4, 2, 3, 1)  # (batch,feat_dim,path_num,len,node_num)
        conv_feat1 = self.feat_conv(feat1)  # (batch,dim,node_num)
        conv_feat2 = self.feat_conv(feat2)  # (batch,dim,node_num)

        if self.use_local_common:
            embedded_struct1 = self.structure_embedding(graph_struct1)  # (batch_size,node_num,path_num,dim)
            embedded_struct1 = embedded_struct1.permute(0, 3, 2, 1)     # (batch_size,dim,path_num,node_num)
            embedded_struct2 = self.structure_embedding(graph_struct2)  # (batch_size,node_num,path_num,dim)
            embedded_struct2 = embedded_struct2.permute(0, 3, 2, 1)     # (batch_size,dim,path_num,node_num)
            conv_struct1 = self.structure_conv(embedded_struct1)        # (batch,dim,node_num)
            conv_struct2 = self.structure_conv(embedded_struct2)        # (batch,dim,node_num)
            graph1_final = self.transform_layer((torch.cat((conv_struct1, conv_feat1), dim=1).permute(0, 2, 1)))
            graph2_final = self.transform_layer((torch.cat((conv_struct2, conv_feat2), dim=1).permute(0, 2, 1)))
        else:
            graph1_final = self.transform_layer(conv_feat1.permute(0, 2, 1))
            graph2_final = self.transform_layer(conv_feat2.permute(0, 2, 1))

        graph1_pooled = self.Global_Max_pool(graph1_final.permute(0, 2, 1)).squeeze()  # (batch,dim)
        graph2_pooled = self.Global_Max_pool(graph2_final.permute(0, 2, 1)).squeeze()  # (batch,dim)
        mlp_out = self.fc(torch.add(graph1_pooled, graph2_pooled))  # (batch,dim)
        out = self.output_layer(mlp_out)
        return out  # (batch_size,class_num)


if __name__ == '__main__':
    para_dic = {
        # 01 embedding layer
        'structure_embedding_number': 54,  # 53+1
        # 02 structure convolution
        'struct_channel_list': [32, 64],  # the first is the structure embedding
        'struct_filter_sizes': [[3, 1]],  # the path conv size
        # 03 feature convolution
        'feat_channel_list': [34, 64],  # the first is feat dimension
        'feat_filter_sizes': [[3, 6, 1]],
        'dropout': 0.25,
        'conv_output_channel': 32,
        'mlp_sizes': [128, 256, 128],   # 89570
        'use_local_common': False,
        # 05 training parameters
        'batch_normalized': True,
        'task_type': 'classification',
        'batch_size': 128,
        'lr': 1e-3,  # learning rate
        'seed': 20,  # random seed
        'epochs': 100,
        'CUDA': True,
        'is_train': True,
        'is_test': False,
        'stop_counter': 200,
        'class_number': 2,
    }
    # print(para_dic)
    # test_model = StructDDI(para_dic)
    # print(test_model)
    # print(model_parameters_calculation(test_model))  # 407842  
    # struct1 = torch.randint(1, 10, (32, 10, 20))
    # struct2 = torch.randint(1, 10, (32, 10, 20))
    # feat1 = torch.randn(32, 10, 20, 6, 34)
    # feat2 = torch.randn(32, 10, 20, 6, 34)
    # out = test_model(struct1, struct2, feat1, feat2)
    # print(out.shape)
