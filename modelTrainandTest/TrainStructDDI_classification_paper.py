import os
import sys

# when run this file in shell, we need the following three line codes
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import pandas as pd
from data.structDDI_dataset import get_three_dataloader
from torch import optim
import time
import torch.nn.functional as F
from model.StructDDI_paper import StructDDI
from util.myutil import *
from sklearn import metrics
import argparse


def get_parameter_from_shell():
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument()


def save_csv(path, data, columns, index):
    df = pd.DataFrame(data * 100, index=index, columns=columns)
    df.to_csv(path, sep=',', float_format='%4.2f')


def test_classification_evaluation(true, pred, class_num=2):
    measure = {}
    true_int_labels = true.astype(int)
    predict_int_labels = np.argmax(pred, 1).astype(int)
    measure['acc'] = metrics.accuracy_score(true_int_labels, predict_int_labels)
    if class_num == 2:
        predict_prob_labels = pred[:, 1]
        measure['f1'] = metrics.f1_score(true_int_labels, predict_int_labels, average='binary')
        # the average parameter will be ignored when y_true is binary＼
        measure['auc'] = metrics.roc_auc_score(true_int_labels, predict_prob_labels)  # binary class auc score
        # the average parameter will be ignored when y_true is binary
        precision, recall, _ = metrics.precision_recall_curve(true_int_labels, predict_prob_labels)
        measure['pr_auc'] = metrics.auc(recall, precision)
        measure['ap'] = metrics.average_precision_score(true_int_labels, predict_prob_labels)
    elif class_num > 2:
        true_prob_labels = value_to_hot(true_int_labels)
        predict_prob_labels = pred
        measure['auc'] = metrics.roc_auc_score(true_prob_labels, predict_prob_labels, average='micro',
                                               multi_class='ovr')
        measure['ap'] = metrics.average_precision_score(true_prob_labels, predict_prob_labels, average='micro')
        measure['recall_macro'] = metrics.recall_score(true_int_labels, predict_int_labels, average='macro')
        measure['precision_macro'] = metrics.precision_score(true_int_labels, predict_int_labels, average='macro')
        measure['f1_macro'] = metrics.f1_score(true_int_labels, predict_int_labels, average='macro')
        measure['ap_micro'] = metrics.average_precision_score(true_prob_labels, predict_prob_labels, average='micro')
    return measure


def train_classification_evaluation(true, pred, class_num=2):
    measure = {}
    true_int_labels = true.astype(int)
    predict_int_labels = np.argmax(pred, 1).astype(int)
    measure['acc'] = metrics.accuracy_score(true_int_labels, predict_int_labels)
    if class_num == 2:
        predict_prob_labels = pred[:, 1]
        measure['f1'] = metrics.f1_score(true_int_labels, predict_int_labels, average='binary')
        # the average parameter will be ignored when y_true is binary＼
        measure['auc'] = metrics.roc_auc_score(true_int_labels, predict_prob_labels)  # binary class auc score
        # the average parameter will be ignored when y_true is binary
        precision, recall, _ = metrics.precision_recall_curve(true_int_labels, predict_prob_labels)
        measure['pr_auc'] = metrics.auc(recall, precision)
        measure['ap'] = metrics.average_precision_score(true_int_labels, predict_prob_labels)
    elif class_num > 2:
        true_prob_labels = value_to_hot(true_int_labels)
        predict_prob_labels = pred
        # measure['auc'] = metrics.roc_auc_score(true_prob_labels, predict_prob_labels, average='micro',
        #                                        multi_class='ovr')
        # the average parameter will be ignored when y_true is binary
        # measure['ap'] = metrics.average_precision_score(true_prob_labels, predict_prob_labels, average='micro')
        # measure['recall_macro'] = metrics.recall_score(true_int_labels, predict_int_labels, average='macro')
        # measure['precision_macro'] = metrics.precision_score(true_int_labels, predict_int_labels, average='macro')
        # measure['f1_macro'] = metrics.f1_score(true_int_labels, predict_int_labels, average='macro')
        # measure['ap_micro'] = metrics.average_precision_score(true_prob_labels, predict_prob_labels, average='micro')
    return measure


def predicting(model, loader, use_gpu=True):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.LongTensor()
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            struct1 = batch_data[0].long()
            struct2 = batch_data[1].long()
            feat1 = batch_data[2].float()
            feat2 = batch_data[3].float()
            label = batch_data[4].long()
            if use_gpu:
                struct1 = struct1.cuda()
                struct2 = struct2.cuda()
                feat1 = feat1.cuda()
                feat2 = feat2.cuda()
                label = label.cuda()
            output = model(struct1, struct2, feat1, feat2)
            # pred_prob_labels
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            # true_int_labels
            total_labels = torch.cat((total_labels, label.cpu()), 0)
    return total_labels.numpy(), total_preds.numpy()


def train_model(model, dataloader, epoch, optimizer, use_gpu=True):
    interval = 100000
    total_preds = torch.Tensor()
    total_labels = torch.LongTensor()
    loss_sum = 0
    num = 0
    model.train()
    for batch_idx, batch_data in enumerate(dataloader):
        struct1 = batch_data[0].long()
        struct2 = batch_data[1].long()
        feat1 = batch_data[2].float()
        feat2 = batch_data[3].float()
        label = batch_data[4].long()
        if use_gpu:
            struct1 = struct1.cuda()
            struct2 = struct2.cuda()
            feat1 = feat1.cuda()
            feat2 = feat2.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        output = model(struct1, struct2, feat1, feat2)
        loss_train = F.nll_loss(output, label)  # classification
        loss_train.backward()
        optimizer.step()
        loss_sum += loss_train.item()
        num += 1
        with torch.no_grad():
            # pred_prob_labels
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            # true_int_labels
            total_labels = torch.cat((total_labels, label.cpu()), 0)

        # if batch_idx % interval == 0:
        #     print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
        #                                                                    batch_idx * 256,
        #                                                                    len(dataloader.dataset),
        #                                                                    100. * batch_idx / len(dataloader),
        #                                                                    loss_train.item()))
    train_labels_array, train_predict_array = total_labels.numpy(), total_preds.numpy()
    loss_value = loss_sum / num
    return loss_value, train_labels_array, train_predict_array


# argv = parse_arguments(sys.argv[1:])
def parse_arguments_text_cci(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default='600')
    parser.add_argument('--lr', type=float, default='1e-3')
    parser.add_argument('--dropout', type=float, default='0.25')
    parser.add_argument('--batch', type=int, default='128')

    parser.add_argument('--conv_output_channel', type=int, default='32')
    parser.add_argument('--struct_channel_list', type=str, default='[32, 128]')
    parser.add_argument('--feat_channel_list', type=str, default='[46, 128]')
    parser.add_argument('--struct_filter_sizes', type=str, default='[[3, 1]]')
    parser.add_argument('--feat_filter_sizes', type=str, default='[[3, 3, 1]]')
    parser.add_argument('--mlp_sizes', type=str, default='[128, 256, 128]')

    parser.add_argument('--data_id', type=int, default='0')
    parser.add_argument('--path_number', type=int, default='5')
    parser.add_argument('--path_length', type=int, default='10')
    parser.add_argument('--max_number', type=int, default='50')
    parser.add_argument('--use_local_common', type=int, default='0')
    parser.add_argument('--seed', type=int, default='0')
    args = parser.parse_args(argv)
    return args


def str_list_to_int_list(str_list):
    str1 = str_list[1: -1].split(',')
    int_list = [int(i) for i in str1]
    return int_list


def change_para_dic_text_cci(parser, para_dic):
    # str_list_of_list_to_int_list
    para_dic['lr'] = parser.lr
    para_dic['epochs'] = parser.epochs
    para_dic['dropout'] = parser.dropout
    para_dic['batch'] = parser.batch

    para_dic['conv_output_channel'] = parser.conv_output_channel
    para_dic['struct_channel_list'] = str_list_to_int_list(parser.struct_channel_list)
    para_dic['feat_channel_list'] = str_list_to_int_list(parser.feat_channel_list)
    para_dic['struct_filter_sizes'] = str_list_of_list_to_int_list(parser.struct_filter_sizes)
    para_dic['feat_filter_sizes'] = str_list_of_list_to_int_list(parser.feat_filter_sizes)
    para_dic['mlp_sizes'] = str_list_to_int_list(parser.mlp_sizes)

    para_dic['data_id'] = parser.data_id
    para_dic['path_number'] = parser.path_number
    para_dic['path_length'] = parser.path_length
    para_dic['max_number'] = parser.max_number
    para_dic['use_local_common'] = parser.use_local_common
    para_dic['seed'] = parser.seed


def setup_seed_without_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # print("fix random seed !")


if __name__ == "__main__":
    path_lists = []
    dataset_name = ['biosnap_raw', 'ddi_binary_raw', 'cci900_raw']
    class_number_list = [2, 2, 2]
    path = "../data/"
    suffix_list = [
        "_index_adj.pzip",  # <list>    # each element is array
        "_index_feat.pzip",
        "_interaction_cleaned_indexed.pzip"  # <list>    # each element is [5304, 4981, 1.0]
    ]
    for name in dataset_name:
        path_lists.append([path + name + '/' + name + suffix for suffix in suffix_list])
    para_dic = {
        'struct_channel_list': [32, 128],  # the first is the structure embedding,
        'struct_filter_sizes': [[3, 1]],  # (path_number, node_number)
        # 03 feature convolution          
        'feat_channel_list': [46, 128],     # the first is feat dimension
        'feat_filter_sizes': [[3, 3, 1]],  # (path_number,path_length,node_number)
        'dropout': 0.25,
        'conv_output_channel': 32,
        'mlp_sizes': [128, 256, 128],
        # 05 training parameters
        'data_id': 0,
        'task_type': 'classification',
        'batch': 128,
        'lr': 1e-3,  # learning rate
        'seed': 20,
        # random seed
        # 或者 save the extraction features including struct pattern array and feat array
        'epochs': 500,              
        'CUDA': True,
        'is_train': True,
        'is_test': True,
        'stop_counter': 50,
        'class_number': 2,
        # 06 structure extraction    
        'path_number': 5,
        'path_length': 10,
        'use_local_common': True,
    }
    print(para_dic)
    # parser = parse_arguments_text_cci(sys.argv[1:])
    # change_para_dic_text_cci(parser, para_dic)
    data_id = para_dic['data_id']
    para_dic['class_number'] = class_number_list[data_id]
    print(dataset_name[data_id])
    setup_seed(para_dic['seed'])
    best_valid_dic = {}
    divide_list = [
        [0.70, 0.10, 0.20],  # biosnap
        [0.70, 0.10, 0.20],  # ddi
        [0.72, 0.18, 0.10]  # cci900
    ]

    train_loader, valid_loader, test_loader, para_dic['structure_embedding_number'] = \
        get_three_dataloader(data_path_list=path_lists[data_id],
                             divide_list=divide_list[data_id],
                             batch_size=para_dic['batch'],
                             path_number=para_dic['path_number'],
                             path_length=para_dic['path_length'],
                             max_node_number=para_dic['max_number'],
                             )
    print(para_dic)
    if para_dic['is_train']:
        model = StructDDI(para_dic)
        optimizer = optim.Adam(model.parameters(), lr=para_dic['lr'])
        # torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        if para_dic['CUDA']:
            model.cuda()
        best_auc = 0
        counter = 0
        start_time = time.time()

        for epoch in range(para_dic['epochs']):
            loss, train_truth, train_predict = train_model(model, train_loader, epoch, optimizer)
            train_results = train_classification_evaluation(train_truth, train_predict, para_dic['class_number'])
            truth, predict = predicting(model, valid_loader)
            valid_results = train_classification_evaluation(truth, predict, para_dic['class_number'])
            duration = time.time() - start_time
            print('Epoch %d Train loss %f (%.3f sec) acc %f auc %f f1 %f Valid: acc %f auc %f f1 %f'
                  % (
                      epoch,
                      loss,
                      duration,
                      train_results['acc'],
                      train_results['auc'],
                      train_results['f1'],
                      valid_results['acc'],
                      valid_results['auc'],
                      valid_results['f1'],
                  ))
            if valid_results['acc'] > best_auc and epoch >= 5:  
                best_auc = valid_results['acc']
                best_valid_dic = valid_results
                counter = 0
                save_dir = '../savedModel/savedStructDDI_paper/' + str(dataset_name[data_id]) + '/'  # saved model
                assert_dir_exist(save_dir)
                print("new model saved")
                save_best_model(model, model_dir=save_dir, best_epoch=epoch)
            else:
                counter += 1

            start_time = time.time()
            if counter == para_dic['stop_counter']:
                print("early stop at epoch %d" % epoch)
                break
    # =================================================================================================
    # evaluate_index = ['auc', 'acc', 'recall_macro', 'precision_macro', 'f1_macro', 'ap_micro']
    if para_dic['is_test']:
        test_model = StructDDI(para_dic)
        if para_dic['CUDA']:
            test_model.cuda()
        load_dir = '../savedModel/savedStructDDI_paper/' + str(dataset_name[data_id]) + '/'  # saved model
        files = glob.glob(load_dir + '*.pkl')
        for file in files:
            epoch_num = file.split('/')[-1].split('.')[0]  # achieve the epoch number
        load_file_dir = load_dir + epoch_num + '.pkl'
        print('load saved model of epoch %d successfully!!!' % int(epoch_num))
        test_model.load_state_dict(torch.load(load_file_dir))
        truth, predict = predicting(test_model, test_loader, use_gpu=True)
        results = test_classification_evaluation(truth, predict, para_dic['class_number'])
        print('Test: auc_micro %f'
              ' acc %f, f1 %f, pr_auc %f' % (
                  results['auc'],
                  results['acc'],
                  results['f1'],
                  results['pr_auc'],
              ))
        res_path = '../result/classification/StructDDI_paper/' + dataset_name[data_id] + '/'
        assert_dir_exist(res_path)
        filename = get_file_name_from_dic(para_dic)
        result_list = []
        for tmp in results.keys():
            result_list.append(results[tmp])
        windows_file_name = dataset_name[data_id] + '_results.txt'
        write_to_txt(file_path=res_path + windows_file_name, content_str=str(results['acc'] * 100))
        write_to_txt(file_path=res_path + windows_file_name, content_str=str(results))
        write_to_txt(file_path=res_path + windows_file_name, content_str=str(para_dic))
        del_file(load_dir)

