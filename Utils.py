import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime
import pickle
import os
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import Dataset, DataLoader
from snfpy import snf
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import random
def precision_k(label_test, label_predict, k):
    num_pos = 100
    num_neg = 100
    label_test = np.argsort(label_test, axis=1)
    label_predict = np.argsort(label_predict, axis=1)
    precision_k_neg = []
    precision_k_pos = []
    neg_test_set = label_test[:, :num_neg]
    pos_test_set = label_test[:, -num_pos:]
    neg_predict_set = label_predict[:, :k]
    pos_predict_set = label_predict[:, -k:]
    for i in range(len(neg_test_set)):
        neg_test = set(neg_test_set[i])
        pos_test = set(pos_test_set[i])
        neg_predict = set(neg_predict_set[i])
        pos_predict = set(pos_predict_set[i])
        precision_k_neg.append(len(neg_test.intersection(neg_predict)) / k)
        precision_k_pos.append(len(pos_test.intersection(pos_predict)) / k)
    return np.mean(precision_k_neg), np.mean(precision_k_pos)
def correlation(label_test, label_predict, correlation_type):
    if correlation_type == 'pearson':
        corr = pearsonr
    elif correlation_type == 'spearman':
        corr = spearmanr
    else:
        raise ValueError("Unknown correlation type: %s" % correlation_type)
    score = []
    for lb_test, lb_predict in zip(label_test, label_predict):
        score.append(corr(lb_test, lb_predict)[0])
    return np.mean(score), score

def train(model, loader, criterion, opt, device):
    model.train()


    for idx, data in enumerate(tqdm(loader, desc='Iteration')):  # tqdm是进度条  返回 enumerate(枚举) 对象。
        # (self.gene, pert_idose_feature, self.drug.loc[self.drug_name[index]].values, self.cell.loc[self.Cell_line_name[index]].values, self.value[index], self.fused_network.loc[self.drug_name[index]].values)

        gene, pert_idose_feature, drug, cell, label,fusion= data
#drug_features, cell_features, device,fused_network, input_gene, input_pert_idose
        output = model(drug, cell,device,fusion, gene, pert_idose_feature)
        loss = criterion(output, label.float().to(device))

        opt.zero_grad()
        loss.backward()
        opt.step()

    print('Train Loss:{}'.format(loss))
    return loss


def validate(model, loader, device):
    # rmse, _, _, _ = validate(model, val_loader, args.device)
    model.eval()

    y_true = []
    y_pred = []
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(loader, desc='Iteration'):
            gene, pert_idose_feature, drug, cell, label, fusion = data
            # drug_features, cell_features, device,fused_network, input_gene, input_pert_idose
            output = model(drug, cell, device, fusion, gene, pert_idose_feature)
            total_loss += F.mse_loss(output, label.float().to(device), reduction='sum')
            y_true.append(label)
            y_pred.append(output)
    y_true1 = y_true
    y_pred1 = y_pred
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    rmse = torch.sqrt(total_loss / len(loader.dataset))
    MAE = mean_absolute_error(y_true.cpu(), y_pred.cpu())
    r2 = r2_score(y_true.cpu(), y_pred.cpu())
    r = pearsonr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]
    spearman= spearmanr(y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())[0]
    precision_neg, precision_pos = precision_k(y_true.cpu().numpy(), y_pred.cpu().numpy(), 100)


    return rmse, MAE, r2, r, spearman, precision_neg, precision_pos , y_true1, y_pred1

class EarlyStopping():
    # EarlyStopping(mode='lower', patience=args.patience)
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            dt = datetime.datetime.now()
            folder = os.path.join(os.getcwd(), 'results')
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder, 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])


def Chebyshev_Distance(matrix):
    sim = np.zeros((len(matrix), len(matrix)))
    for A in range(len(matrix)):
        for B in range(len(matrix)):
            sim[A][B] = np.linalg.norm(matrix[A]-matrix[B],ord=np.inf)

    return sim


class MyDataset(Dataset):
    def __init__(self, gene, drug_dict, cell_dict,fused_network, IC):
        super(MyDataset, self).__init__()
        self.drug, self.cell = drug_dict, cell_dict
        self.fused_network = fused_network
        self.gene = gene.values
        # IC.reset_index(drop=True, inplace=True)  # train_test_split之后，数据集的index混乱，需要reset
        self.drug_name = IC.iloc[:, 0].values
        self.Cell_line_name = IC.iloc[:, 2].values
        self.dosage = IC.iloc[:, 3].values
        self.pert_idose_set = sorted(list(set(self.dosage)))
        self.pert_idose_dict = dict(zip(self.pert_idose_set, list(range(len(self.pert_idose_set)))))

        self.value = IC.loc[:,gene.index.values].values
    def __len__(self):
        return len(self.value)
    def __getitem__(self, index):
        pert_idose_feature = np.zeros(len(self.pert_idose_set))
        pert_idose_feature[self.pert_idose_dict[self.dosage[index]]] = 1
        # self.cell[self.Cell_line_name[index]].adj_t = SparseTensor(row=self.edge_index[0], col=self.edge_index[1])
        return (self.gene, pert_idose_feature, self.drug.loc[self.drug_name[index]].values, self.cell.loc[self.Cell_line_name[index]].values, self.value[index], self.fused_network.loc[self.drug_name[index]].values)

def load_data(args, drug_representation):
    rawdata_dir = args.rawpath
    final_sample = pd.read_csv(args.rawpath + 'data/leve_5_New2.csv', index_col=0)
    train_set, val_set = train_test_split(final_sample, test_size=0.4, random_state=42)
    val_set, test_set = train_test_split(val_set, test_size=0.5, random_state=42)
    drug_features, cell_features, fused_network, Cell_name, Drug_name = read_raw_data(rawdata_dir)
    cell_features_matrix = cell_features[0]
    for i in range(1, len(cell_features)):
        cell_features_matrix = np.hstack((cell_features_matrix, cell_features[i]))

    Drug_ID = pd.read_csv(rawdata_dir + 'data/data/drug.txt', index_col=0, header=None)
    drug_df = pd.DataFrame(drug_representation)
    drug_df.index = [x.lower() for x in Drug_ID.index.values.tolist()]
    merged_A_E = pd.read_csv(rawdata_dir + 'data/Drug_A_E/merged_A_E.csv', index_col=0)
    for idx in drug_df.index:
        if idx in merged_A_E.index:
            merged_A_E.loc[idx] = drug_df.loc[idx].values.tolist()
    merged_A_E.index = Drug_name
    drug_features_matrix = merged_A_E
    gene = pd.read_csv(rawdata_dir + 'data/gene_vector.csv', index_col=0, header=None)
    ############################# del
    del merged_A_E
    del drug_df
    del drug_features
    del cell_features
    ###############################
    Dataset = MyDataset
    train_dataset = Dataset(gene, drug_features_matrix, cell_features_matrix, fused_network, train_set)
    test_dataset = Dataset(gene, drug_features_matrix, cell_features_matrix, fused_network, test_set)
    val_dataset = Dataset(gene, drug_features_matrix, cell_features_matrix, fused_network, val_set)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, val_loader  #


def read_raw_data(rawdata_dir):

    ###drug
    Drug_A1 = pd.read_csv(rawdata_dir+'data/Drug_A_E/A1_Known.csv' , index_col= 0,header=None)
    Drug_A2 = pd.read_csv(rawdata_dir+'data/Drug_A_E/A2_Known.csv' , index_col= 0,header=None)
    Drug_A3 = pd.read_csv(rawdata_dir+'data/Drug_A_E/A3_Known.csv' , index_col= 0,header=None)
    Drug_A4 = pd.read_csv(rawdata_dir+'data/Drug_A_E/A4_Known.csv' , index_col= 0,header=None)
    Drug_A5 = pd.read_csv(rawdata_dir+'data/Drug_A_E/A5_Known.csv' , index_col= 0,header=None)
    Drug_A = pd.concat([Drug_A1, Drug_A2, Drug_A3, Drug_A4, Drug_A5], axis=1)

    Drug_B1 = pd.read_csv(rawdata_dir+'data/Drug_A_E/B1_Known.csv' , index_col= 0,header=None)
    Drug_B2 = pd.read_csv(rawdata_dir+'data/Drug_A_E/B2_Known.csv' , index_col= 0,header=None)
    Drug_B3 = pd.read_csv(rawdata_dir+'data/Drug_A_E/B3_Known.csv' , index_col= 0,header=None)
    Drug_B4 = pd.read_csv(rawdata_dir+'data/Drug_A_E/B4_Known.csv' , index_col= 0,header=None)
    Drug_B5 = pd.read_csv(rawdata_dir+'data/Drug_A_E/B5_Known.csv' , index_col= 0,header=None)
    Drug_B = pd.concat([Drug_B1, Drug_B2, Drug_B3, Drug_B4, Drug_B5], axis=1)

    Drug_C1 = pd.read_csv(rawdata_dir+'data/Drug_A_E/C1_Known.csv' , index_col= 0,header=None)
    Drug_C2 = pd.read_csv(rawdata_dir+'data/Drug_A_E/C2_Known.csv' , index_col= 0,header=None)
    Drug_C3 = pd.read_csv(rawdata_dir+'data/Drug_A_E/C3_Known.csv' , index_col= 0,header=None)
    Drug_C4 = pd.read_csv(rawdata_dir+'data/Drug_A_E/C4_Known.csv' , index_col= 0,header=None)
    Drug_C5 = pd.read_csv(rawdata_dir+'data/Drug_A_E/C5_Known.csv' , index_col= 0,header=None)
    Drug_C = pd.concat([Drug_C1, Drug_C2, Drug_C3, Drug_C4, Drug_C5], axis=1)

    Drug_D1 = pd.read_csv(rawdata_dir+'data/Drug_A_E/D1_Known.csv' , index_col= 0,header=None)
    Drug_D2 = pd.read_csv(rawdata_dir+'data/Drug_A_E/D2_Known.csv' , index_col= 0,header=None)
    Drug_D3 = pd.read_csv(rawdata_dir+'data/Drug_A_E/D3_Known.csv' , index_col= 0,header=None)
    Drug_D4 = pd.read_csv(rawdata_dir+'data/Drug_A_E/D4_Known.csv' , index_col= 0,header=None)
    Drug_D5 = pd.read_csv(rawdata_dir+'data/Drug_A_E/D5_Known.csv' , index_col= 0,header=None)
    Drug_D = pd.concat([Drug_D1, Drug_D2, Drug_D3, Drug_D4, Drug_D5], axis=1)

    Drug_E1 = pd.read_csv(rawdata_dir+'data/Drug_A_E/E1_Known.csv' , index_col= 0,header=None)
    Drug_E2 = pd.read_csv(rawdata_dir+'data/Drug_A_E/E2_Known.csv' , index_col= 0,header=None)
    Drug_E3 = pd.read_csv(rawdata_dir+'data/Drug_A_E/E3_Known.csv' , index_col= 0,header=None)
    Drug_E4 = pd.read_csv(rawdata_dir+'data/Drug_A_E/E4_Known.csv' , index_col= 0,header=None)
    Drug_E5 = pd.read_csv(rawdata_dir+'data/Drug_A_E/E5_Known.csv' , index_col= 0,header=None)
    Drug_E = pd.concat([Drug_E1, Drug_E2, Drug_E3, Drug_E4, Drug_E5], axis=1)
    Drug_name = list(Drug_A.index.values)

    Drug_A_sim =  Chebyshev_Distance(Drug_A.to_numpy())
    Drug_B_sim =  Chebyshev_Distance(Drug_B.to_numpy())
    Drug_C_sim =  Chebyshev_Distance(Drug_C.to_numpy())
    Drug_D_sim =  Chebyshev_Distance(Drug_D.to_numpy())
    Drug_E_sim =  Chebyshev_Distance(Drug_E.to_numpy())

    ###cell
    Cell_mu = pd.read_csv(rawdata_dir+'data/Cell/CCLE_mu.csv' , index_col= 0)
    Cell_cn = pd.read_csv(rawdata_dir+'data/Cell/Cell_cn.csv' , index_col= 0)
    Cell_exp = pd.read_csv(rawdata_dir+'data/Cell/Cell_exp.csv' , index_col= 0)
    Cell_name = list(Cell_mu.index.values)

    drug_features, cell_features = [], []

    drug_features.append(Drug_A_sim)
    drug_features.append(Drug_B_sim)
    drug_features.append(Drug_C_sim)
    drug_features.append(Drug_D_sim)
    drug_features.append(Drug_E_sim)

    cell_features.append(Cell_mu)
    cell_features.append(Cell_cn)
    cell_features.append(Cell_exp)

    fused_network = snf.snf(drug_features, K=20) #K parameter typically represents the number of nearest neighbors considered in the network construction.


    return drug_features, cell_features, fused_network, Cell_name, Drug_name
