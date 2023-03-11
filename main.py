# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 15:06:23 2022

@author: Zuoxibing
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
import time
from sklearn import preprocessing
import torch
import os
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import argparse
import sys

sys.path.append('./networks')
from networks import  models
sys.path.append('./utils')
from utils import graphs_generated
from utils import basics

parser = argparse.ArgumentParser(description='GLFAF')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--data_name', type=str, default='SA', help='PU/SA/LongKou')
parser.add_argument('--scale', type=int, default=400, help='scale')
parser.add_argument('--compactness', type=float, default=1, help='parameter of slic')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='scale')
parser.add_argument('--max_epoch', type=int, default=1000, help='num of training epochs')
parser.add_argument('--hidden_size', type=int, default=64, help='begin hidden_size of network (C1)')
parser.add_argument('--method', type=str, default='GLFAF', help='None')

args = parser.parse_args()
device = torch.device('cuda', args.gpu)
data_name = args.data_name
Scale = args.scale
compactness = args.compactness
learning_rate = args.learning_rate
max_epoch = args.max_epoch
hidden_size = args.hidden_size
method = args.method

file_name = "./result/" + data_name +"_" + str(method) + "_hidden" + str(hidden_size) + "_Scale" + str(Scale) + "/"

num_per_class_list = [50]
Seed_List = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  #

for num_per_class in num_per_class_list:
    result_path = file_name + str(num_per_class)
    print("result_path:", result_path)
    basics.isexists_dir_Create(result_path)

    torch.cuda.empty_cache()
    max_test_acc = 0
    total_param = 0
    FLOPs = 0
    Train_time_all = []
    Test_time_all = []
    OA_all = []
    AA_all = []
    Kappa_all = []
    CA_all = []

    # load and process dataset
    data, gt, class_count = basics.get_HSI(data_name)
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [height, width, bands])

    for index, curr_seed in enumerate(Seed_List):
        gt_reshape = np.reshape(gt, [-1])
        train_data_index, val_data_index, test_data_index = basics.get_train_id_radomsample(gt_reshape, num_per_class, curr_seed)
        train_samples_gt, test_samples_gt, val_samples_gt, train_samples_gt_onehot, test_samples_gt_onehot, val_samples_gt_onehot, train_label_mask, test_label_mask, val_label_mask, Test_GT = basics.get_gt_onehot_mask(
            gt_reshape, height, width, class_count, train_data_index, val_data_index, test_data_index)

        ls = graphs_generated.LDA_SLIC(result_path, data, np.reshape(train_samples_gt, [height, width]),
                                       class_count - 1, compactness)
        ls_time_begin = time.clock()
        Q, S, A, Seg = ls.simple_superpixel(scale=Scale)
        ls_time_end = time.clock()
        LDA_SLIC_Time = ls_time_end - ls_time_begin
        print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))

        Q = torch.from_numpy(Q).to(device)
        A = torch.from_numpy(A).to(device)
        train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
        val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
        train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
        test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
        val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
        train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
        val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

        net_input = np.array(data, np.float32)
        net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)

        # model
        net = models.GLFAF(height, width, bands, class_count, Q, A, hidden_size, device)
        net.to(device)

        total_param = sum([param.nelement() for param in net.parameters()])
        print("total_param: {:.2f}M".format(total_param / 1e6))
        FLOPs = FlopCountAnalysis(net, net_input)
        print("FLOPs:{}G".format(FLOPs.total()/1e9))

        # Training
        zeros = torch.zeros([height * width]).to(device).float()
        train_time_begin = time.clock()
        min_loss = 10000
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        net.train()
        train_cost = []
        val_cost = []
        train_acc = []
        val_acc = []
        for epoch in range(max_epoch + 1):
            optimizer.zero_grad()
            output = net(net_input)
            loss = basics.compute_loss(output, train_samples_gt_onehot, train_label_mask)
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            if epoch % 1 == 0:
                with torch.no_grad():
                    net.eval()
                    output = net(net_input)
                    trainloss = basics.compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = basics.train_eval_metrics(output, train_samples_gt, train_samples_gt_onehot, zeros)
                    valloss = basics.compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = basics.train_eval_metrics(output, val_samples_gt, val_samples_gt_onehot, zeros)
                    print("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (
                    epoch, trainloss, valloss, trainOA, valOA))
                    train_cost.append(trainloss.cpu())
                    train_acc.append(trainOA.cpu())
                    val_cost.append(valloss.cpu())
                    val_acc.append(valOA.cpu())

                    if valloss < min_loss:
                        min_loss = valloss
                        basics.isexists_dir_Create(result_path + "/Model")
                        torch.save(net.state_dict(), result_path + "/Model/model_index_" + str(index) + ".pt")
                        print("saving model...")
                torch.cuda.empty_cache()
                net.train()
        train_time_end = time.clock()
        train_time = train_time_end - train_time_begin + LDA_SLIC_Time
        Train_time_all.append(train_time)

        print("====================training done. starting evaluation...========================\n")
        test_time_begin = time.clock()
        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load(result_path + "/Model/model_index_" + str(index) + ".pt"))
            net.eval()
            output = net(net_input)

            testloss = basics.compute_loss(output, test_samples_gt_onehot, test_label_mask)
            test_OA, test_AA, test_Kappa, test_CA = basics.test_metrics(output, test_samples_gt, test_samples_gt_onehot,
                                                                        Test_GT, zeros, height, width, class_count)
            print("round{}".format(index + 1))
            print("The test_OA is %f" % (test_OA))
            print("The test_AA is %f" % (test_AA))
            print("The test_Kappa is %f" % (test_Kappa))
            print("The test_CA is:", test_CA)

            OA_all.append(test_OA)
            AA_all.append(test_AA)
            Kappa_all.append(test_Kappa)
            CA_all.append(test_CA)
            test_time_end = time.clock()
            Test_time_all.append(test_time_end - test_time_begin)
            print("====================evaluation end...========================\n")

            train_cost = np.squeeze(train_cost)
            val_cost = np.squeeze(val_cost)
            train_acc = np.squeeze(train_acc)
            val_acc = np.squeeze(val_acc)

            if test_OA > max_test_acc:
                max_test_acc = test_OA
                train_cost_best = train_cost
                val_cost_best = val_cost
                train_acc_best = train_acc
                val_acc_best = val_acc
                preds_best = output
        torch.cuda.empty_cache()
        del net

    f = open(result_path + "/train_cost.txt", 'w')
    for i in range(train_cost_best.shape[0]):
        f.write(str(train_cost_best[i]) + '\n')
    f.close()

    f = open(result_path + "/val_cost.txt", 'w')
    for i in range(val_cost_best.shape[0]):
        f.write(str(val_cost_best[i]) + '\n')
    f.close()

    f = open(result_path + "/train_acc.txt", 'w')
    for i in range(train_acc_best.shape[0]):
        f.write(str(train_acc_best[i]) + '\n')
    f.close()

    f = open(result_path + "/val_acc.txt", 'w')
    for i in range(val_acc_best.shape[0]):
        f.write(str(val_acc_best[i]) + '\n')
    f.close()

    predict = torch.argmax(preds_best, 1).cpu().numpy().reshape(gt.shape)
    np.save(result_path + "/" + str(data_name)+"_"+str(method)+"_"+str(num_per_class)+"_prediction.npy", predict)
    
    # 绘分类图
    basics.classification_map_with_back(pred=predict, ground_truth=gt, dpi=600, save_path=result_path + "/" + "whole_" + str(data_name) + "_" + str(num_per_class), data_name = data_name)
    Train_time_all = np.array(Train_time_all)
    Test_time_all = np.array(Test_time_all)
    OA_all = np.array(OA_all)
    AA_all = np.array(AA_all)
    Kappa_all = np.array(Kappa_all)
    CA_all = np.array(CA_all)

    print("######################Final result#########################")
    print("\nnum_per_class={}".format(num_per_class),
          "\n==============================================================================")
    print("total_param：{} M".format(total_param / 1e6))
    print("FLOPs:{} G".format(FLOPs.total()/ 1e9))
    print("Average training time:{}".format(np.mean(Train_time_all)))
    print("Average testing time:{}".format(np.mean(Test_time_all)))

    print("OA for each iter = ", OA_all)
    print("AA for each iter = ", AA_all)
    print("Kappa for each iter = ", Kappa_all)

    print('OA mean+-std =', np.mean(OA_all), '+-', np.std(OA_all))
    print('AA mean+-std =', np.mean(AA_all), '+-', np.std(AA_all))
    print('Kappa mean+-std =', np.mean(Kappa_all), '+-', np.std(Kappa_all))
    print('CA mean+-std =', np.mean(CA_all, 0), '+-', np.std(CA_all, 0))

    f = open(result_path + "/metrics.txt", 'w', encoding='utf-8')
    f.write("Method：" + str(method) + '\n')
    f.write("seeds：" + str(Seed_List) + '\n')
    f.write(" the number of training samples for each class：" + str(num_per_class) + '\n')
    f.write("total_param：" + str(total_param / 1e6) + 'M' + '\n')
    f.write("FLOPs(G)：" + str((FLOPs.total())/ 1e9) + 'G' + '\n')
    f.write("Average training time：" + str(np.mean(Train_time_all)) + '\n')
    f.write("Average testing time：" + str(np.mean(Test_time_all)) + '\n')
    f.write('\n')

    f.write("OA for each iter：" + str(OA_all) + '\n')
    f.write("AA for each iter：" + str(AA_all) + '\n')
    f.write("Kappa for each itert：" + str(Kappa_all) + '\n')
    f.write('\n')

    f.write("OA (mean+-std)：" + str(np.mean(OA_all)) + '+-' + str(np.std(OA_all)) + '\n')
    f.write("AA (mean+-std)：" + str(np.mean(AA_all)) + '+-' + str(np.std(AA_all)) + '\n')
    f.write("Kappa (mean+-std)：" + str(np.mean(Kappa_all)) + '+-' + str(np.std(Kappa_all)) + '\n')
    f.write("Accuracy of each class (mean+-std)：" + str(np.mean(CA_all, 0)) + '+-' + str(np.std(CA_all, 0)) + '\n')
    f.close()