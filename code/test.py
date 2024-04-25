from tqdm import tqdm
import torch
from torchmetrics import R2Score
import numpy as np
import pandas as pd

# rdkit imports
from rdkit import RDLogger
from rdkit import rdBase
from rdkit import Chem

# torch imports
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

#dgl imports
import dgl

from model import VreactModel
from moleculegraph2 import get_graph_from_smile
from utils import *
from train import get_metrics

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True

setup_seed(32)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()
smoothl1_loss_fn=torch.nn.SmoothL1Loss()
r2_score=R2Score().to(device)


'''
Parameter definition
'''
dropout=0.010727935587392982
t=6
interaction='tanh-general'

'''
dataset
'''
def collate(samples):
    voc_graphs,oxidant_graphs, labels = map(list, zip(*samples))
    voc_graphs = dgl.batch(voc_graphs)
    oxidant_graphs = dgl.batch(oxidant_graphs)
    voc_len_matrix = get_len_matrix(voc_graphs.batch_num_nodes())
    oxidant_len_matrix = get_len_matrix(oxidant_graphs.batch_num_nodes())
    
    return voc_graphs, oxidant_graphs, voc_len_matrix,oxidant_len_matrix, labels


class Dataclass(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        voc = self.dataset.loc[idx]['voc_smiles']
        mol = Chem.MolFromSmiles(voc)
        mol = Chem.AddHs(mol)
        voc = Chem.MolToSmiles(mol)
        voc_graph = get_graph_from_smile(voc)

        oxidant = self.dataset.loc[idx]['Rxn_smiles']
        mol = Chem.MolFromSmiles(oxidant)
        mol = Chem.AddHs(mol)
        oxidant = Chem.MolToSmiles(mol)

        oxidant_graph = get_graph_from_smile(oxidant)
        delta_g = self.dataset.loc[idx]['logk']
        return [voc_graph, oxidant_graph, [delta_g]]

test_df=pd.read_csv('test.txt',sep='\t')

test_dataset=Dataclass(test_df)

test_loader=DataLoader(test_dataset, collate_fn=collate,batch_size=115,shuffle=True)

'''
loading model
'''

model = VreactModel(dropout=dropout,interaction=interaction,num_step_message_passing=t)
model.to(device)
model.load_state_dict(torch.load('Vreact.tar'))
model.eval()

val_loss, mae_loss,r2, outputs,labels= get_metrics(model, test_loader)

preds=pd.DataFrame({'labels':labels,'preds':outputs})
preds.to_csv('test_preds.txt',sep='\t')

print(val_loss,mae_loss,r2)