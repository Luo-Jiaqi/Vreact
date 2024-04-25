import numpy as np

from dgl import DGLGraph
from dgl.nn.pytorch import Set2Set, NNConv, GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F



class GatherModel(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`
    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 42.
    edge_input_dim : int
        Dimension of input edge feature, default to be 10.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 42.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    """

    def __init__(self,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 ):
        super(GatherModel, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.set2set = Set2Set(node_hidden_dim, 2, 1)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(in_feats=node_hidden_dim,
                           out_feats=node_hidden_dim,
                           edge_func=edge_network,
                           aggregator_type='sum',
                           residual=True
                           )

    def forward(self, g, n_feat, e_feat):
        """Returns the node embeddings after message passing phase.
        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.
        Returns
        -------
        res : node features
        """

        init = n_feat.clone()
        out = F.relu(self.lin0(n_feat))
        for i in range(self.num_step_message_passing):
            if e_feat is not None:
                m = torch.relu(self.conv(g, out, e_feat))
            else:
                m = torch.relu(self.conv.bias +  self.conv.res_fc(out))
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init


class VreactModel(nn.Module):
    """
    This the main class for Vreact model
    """

    def __init__(self,
                 dropout,
                 node_input_dim=42,
                 edge_input_dim=10,
                 node_hidden_dim=42,
                 edge_hidden_dim=42,
                 num_step_message_passing=6,
                 interaction='dot',
                 num_step_set2_set=2,
                 num_layer_set2set=1,
                 ):
        super(VreactModel, self).__init__()

        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.dropout=dropout
        self.voc_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                         self.node_hidden_dim, self.edge_input_dim,
                                         self.num_step_message_passing,
                                         )
        self.oxidant_gather = GatherModel(self.node_input_dim, self.edge_input_dim,
                                          self.node_hidden_dim, self.edge_input_dim,
                                          self.num_step_message_passing,
                                          )

        self.fc1 = nn.Sequential(nn.Linear(8 * self.node_hidden_dim, 256),nn.Dropout(p=self.dropout))
        self.fc2 = nn.Sequential(nn.Linear(256, 128),nn.Dropout(p=self.dropout))
        self.fc3 = nn.Sequential(nn.Linear(128, 1),nn.Dropout(p=self.dropout))
        self.imap = nn.Linear(84, 1)

        self.num_step_set2set = num_step_set2_set
        self.num_layer_set2set = num_layer_set2set
        self.set2set_voc = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)
        self.set2set_oxidant = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

    def forward(self, data):
        voc = data[0]
        oxidant = data[1]
        voc_len = data[2]
        oxidant_len = data[3]
        # node embeddings after interaction phase
        voc_features = self.voc_gather(voc, voc.ndata['x'].float(), voc.edata['w'].float())
        try:
            # if edge exists in a molecule
            oxidant_features = self.oxidant_gather(oxidant, oxidant.ndata['x'].float(), oxidant.edata['w'].float())
        except:
            # if edge doesn't exist in a molecule, for example in case of water
            oxidant_features = self.oxidant_gather(oxidant, oxidant.ndata['x'].float(), None)

        # Interaction phase
        len_map = torch.mm(voc_len.t(), oxidant_len)

        if 'dot' not in self.interaction:
            X1 = voc_features.unsqueeze(0)
            Y1 = oxidant_features.unsqueeze(1)
            X2 = X1.repeat(oxidant_features.shape[0], 1, 1)
            Y2 = Y1.repeat(1, voc_features.shape[0], 1)
            Z = torch.cat([X2, Y2], -1)

            if self.interaction == 'general':
                interaction_map = self.imap(Z).squeeze(2)
            if self.interaction == 'tanh-general':
                interaction_map = torch.tanh(self.imap(Z)).squeeze(2)

            interaction_map = torch.mul(len_map.float(), interaction_map.t())
            ret_interaction_map = torch.clone(interaction_map)

        elif 'dot' in self.interaction:
            interaction_map = torch.mm(voc_features, oxidant_features.t())
            if 'scaled' in self.interaction:
                interaction_map = interaction_map / (np.sqrt(self.node_hidden_dim))

            ret_interaction_map = torch.clone(interaction_map)
            ret_interaction_map = torch.mul(len_map.float(), ret_interaction_map)
            interaction_map = torch.tanh(interaction_map)
    
        oxidant_prime = torch.mm(interaction_map.t(), voc_features)
        voc_prime = torch.mm(interaction_map, oxidant_features)

        # Prediction phase
        voc_features = torch.cat((voc_features, voc_prime), dim=1)
        oxidant_features = torch.cat((oxidant_features, oxidant_prime), dim=1)

        voc_features = self.set2set_voc(voc, voc_features)
        oxidant_features = self.set2set_oxidant(oxidant, oxidant_features)

        final_features = torch.cat((voc_features, oxidant_features), 1)
        predictions = torch.relu(self.fc1(final_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        return predictions, ret_interaction_map