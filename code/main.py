# python imports
import pandas as pd
import warnings
import os
import argparse
import numpy

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

import optuna

# local imports
from model import CIGINModel
from train import train
from moleculegraph2 import get_graph_from_smile
from utils import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='cigin', help="The name of the current project: default: CIGIN")
parser.add_argument('--interaction', help="type of interaction function to use: dot | scaled-dot | general | "
                                          "tanh-general", default='tanh-general')
parser.add_argument('--max_epochs', required=False, default=200, help="The max number of epochs for training")
parser.add_argument('--batch_size', required=False, default=115, help="The batch size for training")

args = parser.parse_args()
project_name = args.name
interaction = args.interaction
max_epochs = int(args.max_epochs)
'''
batch_size = int(args.batch_size)
'''

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True

setup_seed(32)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")




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
        delta_g = self.dataset.loc[idx]['log10']
        return [voc_graph, oxidant_graph, [delta_g]]


def main_optim(lr,weight_decay,batch_size,dropout,t):
    
    train_df = pd.read_csv('train.txt',sep='\t')
    valid_df = pd.read_csv('valid.txt',sep='\t')

    train_dataset = Dataclass(train_df)
    valid_dataset = Dataclass(valid_df)

   
    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=128)
    
    model = =VreactModel(dropout=dropout,interaction=interaction,num_step_message_passing=t)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)
    

    val_mse=train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)
    return val_mse


def objective(trial):
    lr=trial.suggest_float('lr',0.0005,0.005)
    batch_size=trial.suggest_int('batch_size',8,256)
    weight_decay=trial.suggest_float('weight_decay',0,0.01)
    dropout=trial.suggest_float('dropout',0,0.5)
    t=trial.suggest_int('T',4,10)
    val_mse=main_optim(lr, weight_decay,batch_size,dropout,t)
    
    return val_mse

study=optuna.create_study(direction='minimize',study_name='vocs_log')
study.optimize(objective,n_trials=100)
print(study.best_params)
print(study.best_trial)
print(study.trials)


def main():
    train_df = pd.read_csv('train.txt', sep="\t")
    valid_df = pd.read_csv('valid.txt', sep="\t")

    train_dataset = Dataclass(train_df)
    valid_dataset = Dataclass(valid_df)

   
    train_loader = DataLoader(train_dataset, collate_fn=collate, batch_size=115, shuffle=True)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=128)
        


    
    model = CIGINModel(interaction=interaction,dropout=0.010727935587392982,num_step_message_passing=6)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004544422129052773,weight_decay=0.0023082051329494867)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, mode='min', verbose=True)
    

    train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, project_name)
    
 
main()



