#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import sys
import os
import math
import pandas as pd
import os

import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import torch
import torch.nn.functional as F
from HGAT.kgutils import build_graph
from HGAT.utils import set_seeds
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
#++++++++
from HGAT.MAGNAKGEModel import KGEModel
from HGAT.lossfunction import MSELoss, BCESmoothLoss
from HGAT.data_preprocess import Data, fold_train_test_idx

import sklearn.metrics as metrics
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]')
    parser.add_argument('--cuda', default='cpu', action='store_true', help='use GPU')
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--nFold', type=int, default=1)
    parser.add_argument('--neg_times', type=int, default=1)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--combine_sim_network', type=bool, default=False)
    parser.add_argument('--data_dir', type=str, default=current_directory+'/data/Drug')
    parser.add_argument('-save', '--save_dir',
                        default='../results/{}/alpha_cal{}_alpha{}_edge_drop{}_hops{}_layers{}_topk{}_head{}_'
                                'feat_fusion_{}_layer_nm{}/',
                        type=str)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="adam second beta value")
    parser.add_argument('--smoothing', default=0.01, type=float, help='smoothing factor')
    parser.add_argument('--regularization', default=1.0, type=float)
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="weight_decay of adam")

    parser.add_argument("--att_drop", type=float, default=0., help="attention drop out")
    parser.add_argument("--input_drop", type=float, default=0., help="input feature drop out")
    parser.add_argument("--fea_drop", type=float, default=0., help="feature drop out")
    parser.add_argument("--loss_type", type=str, default='MSE', help="MSE, BCE, BCEsmooth, MSESmooth")
    parser.add_argument("--topk_type", type=str, default='local', help="top k type, option: local")
    parser.add_argument('--max_steps', default=2, type=int)

    parser.add_argument("--alpha", type=float, default=0.05, help="random walk with restart")
    parser.add_argument("--edge_drop", type=float, default=0.1, help="graph edge drop out")
    parser.add_argument("--top_k", type=int, default=10, help="top k")
    parser.add_argument("--hops", type=int, default=2, help="hop number")
    parser.add_argument("--layers", type=int, default=3, help="number of layers")
    parser.add_argument('--num_heads', default=8, type=int)

    parser.add_argument("--alpha_cal", type=bool, default=1, help="compute alpha dynamically, 0: no, 1: yes")
    parser.add_argument('--layer_feat_fusion', default='average', type=str, help='concatenate, average, max, last')
    parser.add_argument('--residual', default=0, type=int, help='residual connection, 0: no, 1: yes')
    parser.add_argument('--layer_norm', default=1, type=int, help='layer normalization, 0: no, 1: yes')

    parser.add_argument('-cpu', '--cpu_num', default=0, type=int)
    parser.add_argument('-d', '--hidden_dim', default=3200, type=int)
    parser.add_argument('-ee', '--ent_embed_dim', default=3200, type=int)
    parser.add_argument('-er', '--rel_embed_dim', default=256, type=int)
    parser.add_argument('-e', '--embed_dim', default=256, type=int)
    parser.add_argument("--slope", type=float, default=0.2, help="leaky relu slope")
    parser.add_argument("--clip", type=float, default=1.0, help="grad_clip")
    parser.add_argument('--patience', type=int, default=30, help="used for early stop")
    parser.add_argument('--feed_forward', type=int, default=0, help="0: no, 1: yes")

    parser.add_argument("--graph_on", type=int, default=1, help="Using graph")
    parser.add_argument("--trans_on", type=int, default=0, help="Using transformer")
    parser.add_argument("--mask_on", type=int, default=1, help="Using graph")
    parser.add_argument('--project_on', default=1, type=int)
    parser.add_argument('--inverse_relation', default=True, type=bool)
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--self_loop", type=int, default=1, help="self loop")
    parser.add_argument("--only_test", type=bool, default=False, help="only test, no training")

    return parser.parse_args()

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def calcu_metric(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    confusion = confusion_matrix(y_true, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    acc = (TP + TN) / float(TP + TN + FP + FN)
    sensitity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)

    mcc = matthews_corrcoef(y_true, y_pred)
    precision = TP / (TP + FP)  # equal to: precision_score(y_true, y_pred)
    recall = TP / (TP + FN) # equal to: recall_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall) # equal to: f1_score(y_true, y_pred)

    return round(acc, 4), round(sensitity, 4), round(specificity, 4), round(mcc, 4), \
           round(precision, 4), round(recall, 4), round(f1, 4)


def get_au_aupr(y_true, y_score):
    y_score = y_score[:, 1]
    auc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = metrics.auc(recall, precision)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # aupr = average_precision_score(y_true, y_score)

    return round(auc, 4), round(aupr, 4)#, thresholds


def graph_construction(args, nets, idx=None):
    with_cuda = args.cuda

    graph, num_one_dir_edges, num_relation = build_graph(args, net=nets, idx=idx)
    print('Graph information (nodes = {}, edges={})'.format(graph.number_of_nodes(), graph.number_of_edges()))

    if with_cuda:
        for key, value in graph.ndata.items():
            graph.ndata[key] = value.to(args.cuda)
        for key, value in graph.edata.items():
            graph.edata[key] = value.to(args.cuda)
    return graph, num_one_dir_edges, num_relation

def get_loss(y, logits):
    loss = None
    if args.loss_type == 'BCE':
        y_label = torch.LongTensor(y).to(logits.device)
        loss = F.nll_loss(torch.log(logits), y_label)
    elif args.loss_type == 'BCESmooth':
        loss_bce_smooth = BCESmoothLoss(args.smoothing)
        y_label = torch.FloatTensor(y).to(logits.device)
        loss = loss_bce_smooth(y_label, logits)
    elif args.loss_type == 'MSE':
        y_pred, _ = torch.max(torch.sigmoid(logits), dim=1)
        y_truth = torch.FloatTensor(y).to(logits.device)
        y_truth = y_truth.reshape(-1)

        loss = F.mse_loss(y_truth, y_pred)
    elif args.loss_type == 'MSESmooth':
        loss_mse_smooth = MSELoss(args.smoothing)
        y_pred, _ = torch.max(torch.sigmoid(logits), dim=1)
        y_truth = torch.FloatTensor(y).to(logits.device)
        y_truth = y_truth.reshape(-1)
        loss = loss_mse_smooth(y_truth, y_pred)

    return loss

def training(model, graph, optimizer,  netdata):
    model.train()
    output_drug_protein, output_drug_drug, output_drug_disease, output_drug_se, output_protein_disease, output_protein_protein, output_protein_drug, output_se_drug, output_disease_protein, output_disease_drug, drug_embed= model(graph=graph, type_mask=netdata.type_mask, index=None)#train_idx
    # y = netdata.drug_protein#[train_idx[0], train_idx[1]]
    loss1 = get_loss(netdata.drug_protein, output_drug_protein)  # loss
    loss2 = get_loss(netdata.drug_drug, output_drug_drug)
    loss3 = get_loss(netdata.drug_disease, output_drug_disease)
    loss4 = get_loss(netdata.drug_sideEffect, output_drug_se)
    loss5 = get_loss(netdata.protein_disease, output_protein_disease)
    loss6 = get_loss(netdata.protein_protein, output_protein_protein)
    loss7 = get_loss(netdata.protein_drug, output_protein_drug)
    loss8 = get_loss(netdata.sideEffect_drug, output_se_drug)
    loss9 = get_loss(netdata.disease_drug, output_disease_drug)
    loss10 = get_loss(netdata.disease_protein, output_disease_protein)


    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9 + loss10

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return

def test(model, graph, netdata):
    model.eval()
    with torch.no_grad():
        _, output_drug_drug, _, _, _, _, _, _, _, _, drug_embed= model(graph=graph, type_mask=netdata.type_mask, index=None)
        loss = get_loss(netdata.drug_drug, output_drug_drug)
        logits = output_drug_drug.cpu().numpy()
        y_logits = logits[:, 1]

    auc = roc_auc_score(netdata.drug_drug.flatten(), y_logits)
    aupr = average_precision_score(netdata.drug_drug.flatten(), y_logits)
    acc = accuracy_score(y_true=netdata.drug_drug.flatten(), y_pred=np.argmax(logits, axis=1))
    return loss, acc, auc, aupr, drug_embed

def Pre_train(args):
    random_seed = args.seed
    set_seeds(random_seed)
    net_data = Data(args)
    args.nentity = net_data.num_nodes

    graph_train, num_one_dir_edges, num_relation = graph_construction(args, net_data, idx=None)

    args.nrelation = num_relation
    args.nedges = num_one_dir_edges

    model = KGEModel(args, net_data.drug_embedding_initial)
    if args.cuda:
        model = model.to(args.cuda)

    optimizer = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                     betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)


    best_auc = 0

    drug_embed1 = None
    counter = 0


    for epoch in range(args.num_epoch):
        training(model, graph_train, optimizer, net_data)
        # scheduler.step()

        loss, acc, auc, aupr, drug_embed = test(model, graph_train, net_data)
        print('Epoch {:d} | loss {:.6f} | acc {:.4f} | auc {:.4f} | aupr {:.4f}'.format(epoch, loss, acc, auc, aupr))


        # early stopping
        if best_auc < auc:
            best_acc = acc
            best_auc = auc
            best_aupr, drug_embed1 = aupr, drug_embed
            counter = 0
        else:
            counter += 1

        if counter > args.patience:
            print('Early stopping!')
            break
    return drug_embed1.cpu().detach().numpy()



args = parse_args()
drug_representation = Pre_train(args)
np.save(args.data_dir +'drug_representation.npy', drug_representation)
