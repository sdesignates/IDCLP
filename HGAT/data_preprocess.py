import os
import numpy as np
import json


def get_adjM(drug_drug, drug_protein, drug_disease, drug_sideEffect, protein_protein, protein_disease,
             num_drug, num_protein, num_disease, num_se):
    # Drug-0, Protein-1, Disease-2, Side-effect-3
    dim = num_drug + num_protein + num_disease + num_se
    adjM = np.zeros((dim, dim), dtype=int)
    adjM[:num_drug, :num_drug] = drug_drug
    adjM[:num_drug, num_drug: num_drug + num_protein] = drug_protein
    adjM[:num_drug, num_drug + num_protein: num_drug + num_protein + num_disease] = drug_disease
    adjM[:num_drug, num_drug + num_protein + num_disease:] = drug_sideEffect
    adjM[num_drug: num_drug + num_protein, num_drug: num_drug + num_protein] = protein_protein
    adjM[num_drug: num_drug + num_protein,
    num_drug + num_protein: num_drug + num_protein + num_disease] = protein_disease

    adjM[num_drug: num_drug + num_protein, :num_drug] = drug_protein.T
    adjM[num_drug + num_protein: num_drug + num_protein + num_disease, :num_drug] = drug_disease.T
    adjM[num_drug + num_protein + num_disease:, :num_drug] = drug_sideEffect.T
    adjM[num_drug + num_protein: num_drug + num_protein + num_disease,
    num_drug: num_drug + num_protein] = protein_disease.T

    return adjM

def fold_train_test_idx(pos_folds, neg_folds, nFold, foldID):
    train_pos_idx = []
    train_neg_idx = []
    test_fold_idx = []
    for fold in range(nFold):
        if fold == foldID:
            continue
        train_pos_idx.append(pos_folds['fold_' + str(fold)])
        train_neg_idx.append(neg_folds['fold_' + str(fold)])
    train_pos_idx = np.concatenate(train_pos_idx, axis=1)
    train_neg_idx = np.concatenate(train_neg_idx, axis=1)
    train_fold_idx = np.concatenate([train_pos_idx, train_neg_idx], axis=1)

    test_fold_idx.append(pos_folds['fold_' + str(foldID)])
    test_fold_idx.append(neg_folds['fold_' + str(foldID)])
    test_fold_idx = np.concatenate(test_fold_idx, axis=1)
    return train_fold_idx, test_fold_idx

def get_type_mask(num_drug, num_protein, num_disease, num_se):
    # Drug-0, Protein-1, Disease-2, Side-effect-3
    dim = num_drug + num_protein + num_disease + num_se
    type_mask = np.zeros((dim), dtype=int)
    type_mask[num_drug: num_drug + num_protein] = 1
    type_mask[num_drug + num_protein: num_drug + num_protein + num_disease] = 2
    type_mask[num_drug + num_protein + num_disease:] = 3
    return type_mask

def get_type_mask_zheng(num_drug, num_protein, num_st, num_su, num_se, num_go):
    # Drug-0, Protein-1, chemical structures-2, sunstituent-3, Side-effect-4, go terms-5
    dim = num_drug + num_protein + num_st + num_su + num_se + num_go
    type_mask = np.zeros((dim), dtype=int)
    type_mask[num_drug: num_drug + num_protein] = 1
    type_mask[num_drug + num_protein: num_drug + num_protein + num_st] = 2
    type_mask[num_drug + num_protein + num_st: num_drug + num_protein + num_st + num_su] = 3
    type_mask[num_drug + num_protein + num_st + num_su: num_drug + num_protein + num_st + num_su + num_se] = 4
    type_mask[num_drug + num_protein + num_st + num_su + num_se:] = 5
    return type_mask

def Load_Adj_Togerther(dir_lists, ratio=0.01):
    a = np.loadtxt(dir_lists[0])
    print('Before Interactions: ', sum(sum(a)))

    for i in range(len(dir_lists) - 1):
        b_new = np.zeros_like(a)

        b = np.loadtxt(dir_lists[i + 1])
        # remove diagonal elements
        b = b - np.diag(np.diag(b))
        # if the matrix are symmetrical, get the triu matrix
        if (b == b.T).all():
            b = np.triu(b)
        index = np.nonzero(b)
        values = b[index]
        index = np.transpose(index)
        edgelist = np.concatenate([index, values.reshape(-1, 1)], axis=1)
        topK_idx = np.argpartition(edgelist[:, 2], int(ratio * len(edgelist)))[-(int(ratio * len(edgelist))):]
        # print(len(topK_idx))
        select_idx = index[topK_idx]
        b_new[select_idx[:, 0], select_idx[:, 1]] = b[select_idx[:, 0], select_idx[:, 1]]
        a = a + b_new

    a = a + a.T
    a[a > 0] = 1
    a[a <= 0] = 0
    a = a + np.eye(a.shape[0], a.shape[1])
    a = a.astype(int)
    print('After Interactions: ', sum(sum(a)))

    return a

class Data(object):
    def __init__(self, args):
        # interaction network path
        drug_drug_path = args.data_dir + '/mat_drug_drug.txt'
        protein_protein_path = args.data_dir + '/mat_protein_protein.txt'
        drug_protein_path = args.data_dir + '/mat_drug_protein.txt'
        drug_disease_path = args.data_dir + '/mat_drug_disease.txt'
        drug_sideEffect_path = args.data_dir + '/mat_drug_se.txt'
        protein_disease_path = args.data_dir + '/mat_protein_disease.txt'

        self.drug_embedding_initial = np.loadtxt(args.data_dir + '/drug_db_A_E.txt')

        self.drug_protein = np.loadtxt(drug_protein_path, dtype=int)
        self.drug_disease = np.loadtxt(drug_disease_path, dtype=int)
        self.drug_sideEffect = np.loadtxt(drug_sideEffect_path, dtype=int)
        self.protein_disease = np.loadtxt(protein_disease_path, dtype=int)
        self.drug_drug = np.loadtxt(drug_drug_path, dtype=int)
        self.protein_protein = np.loadtxt(protein_protein_path, dtype=int)
        self.protein_drug = self.drug_protein.T
        self.disease_drug = self.drug_disease.T
        self.sideEffect_drug = self.drug_sideEffect.T
        self.disease_protein = self.protein_disease.T

        self.num_drug, self.num_protein = self.drug_protein.shape
        self.num_disease = self.drug_disease.shape[1]
        self.num_se = self.drug_sideEffect.shape[1]
        self.type_mask = get_type_mask(self.num_drug, self.num_protein, self.num_disease, self.num_se)
        self.num_nodes = len(self.type_mask)


