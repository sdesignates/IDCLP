import torch
import torch.nn as nn
import torch.nn.functional as F
from drug_gene_attention import DrugGeneAttention, DrugCellAttention
class IDCLP(nn.Module):
    def __init__(self, num_gene, drugs_dim, cells_dim,gene_input_dim, gene_embed_dim,pert_idose_input_dim, pert_idose_emb_dim, embed_dim, fused_dim, bathsize, dropout1, dropout2):
        super(IDCLP, self).__init__()
        self.drugs_dim = drugs_dim
        self.cells_dim = cells_dim
        self.gene_emb_dim = gene_embed_dim

        self.batchsize = bathsize
        # self.drug_dim = self.drugs_dim//12
        # self.cell_dim = (self.cells_dim-580)//3
        self.num_gene = num_gene

        self.embed_dim = embed_dim
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.gene_embed = nn.Linear(gene_input_dim, gene_embed_dim)
        self.pert_idose_embed = nn.Linear(pert_idose_input_dim, pert_idose_emb_dim)

        self.drugs_layer = nn.Linear(self.drugs_dim, self.embed_dim)
        self.drugs_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.drugs_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.cells_layer = nn.Linear(self.cells_dim, self.embed_dim)
        self.cells_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.cells_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.drug_gene_attn = DrugGeneAttention(gene_embed_dim, self.embed_dim, n_layers=2, n_heads=4, pf_dim=512,
                                                dropout=dropout1, device='cpu')
        self.drug_cell_attn = DrugCellAttention(gene_embed_dim, self.embed_dim, n_layers=2, n_heads=4, pf_dim=512,
                                                dropout=dropout2, device='cpu')

        self.fused_network_layer = nn.Linear(fused_dim, self.embed_dim)
        self.fused_network_bn = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.fused_network_layer_1 = nn.Linear(self.embed_dim, self.embed_dim)

        self.total_layer_emb = gene_embed_dim  + self.embed_dim + self.embed_dim + self.embed_dim + self.pert_idose_embed + self.embed_dim + self.embed_dim
#gene_embed_dim + pert_idose_emb_dim  + gene_embed_dim + gene_embed_dim + self.embed_dim + self.embed_dim + self.embed_dim
#(x_drugs, gene_feature, pert_idose_embed, drug_gene_embed, drug_cell_embed, fused_network, x_cells)
        self.linear_1 = nn.Linear(self.total_layer_emb, 128)
        self.linear_2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout2)
        self.relu = nn.ReLU()



    def forward(self, drug_features, cell_features, device,fused_network, input_gene, input_pert_idose):

        fused_network= F.relu(self.fused_network_bn(self.fused_network_layer(fused_network.float().to(device))), inplace=True)
        fused_network= F.dropout(fused_network, training=self.training, p=self.dropout1)
        fused_network = self.fused_network_layer_1(fused_network)

        x_drugs = F.relu(self.drugs_bn(self.drugs_layer(drug_features.float().to(device))), inplace=True)
        x_drugs = F.dropout(x_drugs, training=self.training, p=self.dropout1)
        x_drugs = self.drugs_layer_1(x_drugs)

        x_cells = F.relu(self.cells_bn(self.cells_layer(cell_features.float().to(device))), inplace=True)
        x_cells = F.dropout(x_cells, training=self.training, p=self.dropout1)
        x_cells = self.cells_layer_1(x_cells)

        gene_feature = self.gene_embed(input_gene.float())

        pert_idose_embed = self.pert_idose_embed(input_pert_idose.float())
        drug_gene_embed, _ = self.drug_gene_attn(gene_feature.float(), x_drugs.float(), None, None)
        drug_cell_embed, _ = self.drug_cell_attn(x_cells.float(), x_drugs.float(), None, None)
##############################################################################################################################################
        x_drugs = x_drugs.unsqueeze(1)
        x_drugs = x_drugs.repeat(1, self.num_gene, 1)
        x_cells = x_cells.unsqueeze(1)
        x_cells = x_cells.repeat(1, self.num_gene, 1)
        fused_network = fused_network.unsqueeze(1)
        fused_network = fused_network.repeat(1, self.num_gene, 1)
        pert_idose_embed = pert_idose_embed.unsqueeze(1)
        pert_idose_embed = pert_idose_embed.repeat(1, self.num_gene, 1)
        drug_cell_embed = drug_cell_embed.unsqueeze(1)
        drug_cell_embed = drug_cell_embed.repeat(1, self.num_gene, 1)

        #(x_drugs, gene_feature, pert_idose_embed, drug_gene_embed, drug_cell_embed, fused_network, x_cells)
        total = torch.cat((x_drugs, gene_feature,  drug_gene_embed,  fused_network), dim=2)
        drug_gene_embed = self.relu(total)
        # drug_gene_embed = [batch * num_gene * (drug_D1_gene_embed + drug_embed + gene_embed + pert_type_embed + cell_id_embed + pert_idose_embed)]
        out = self.linear_1(drug_gene_embed)
        # out = [batch * num_gene * hid_dim]
        out = self.relu(out)
        # out = [batch * num_gene * hid_dim]
        out = self.linear_2(out)
        # out = [batch * num_gene * 1]
        out = out.squeeze(2)
##############################################################################################################################################
        return  out #regression.squeeze()


