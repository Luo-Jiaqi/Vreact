import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import dgl
import matplotlib
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from moleculegraph2 import get_graph_from_smile
from IPython.display import SVG, display
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def draw(smiles, weights):

	g=get_graph_from_smile(smiles)
	c=weights.detach().cpu().numpy().flatten().tolist()
	norm = matplotlib.colors.Normalize(vmin=0,vmax=(sum(c)/len(c)))
	cmap = cm.get_cmap('summer_r')
	plt_colors = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
	atom_colors = {i: plt_colors.to_rgba(weights[i].data.item()) for i in range(g.number_of_nodes())}
	plt_colors._A = [] 
	cb = plt.colorbar(plt_colors)
	cb.set_ticks([])

	mol = Chem.MolFromSmiles(smiles)
	rdDepictor.Compute2DCoords(mol)
	drawer = rdMolDraw2D.MolDraw2DSVG(300,300)

	drawer.SetFontSize(1)
	op = drawer.drawOptions().addAtomIndices=True
	mol = rdMolDraw2D.PrepareMolForDrawing(mol)
	drawer.DrawMolecule(mol,highlightAtoms=range(g.number_of_nodes()),highlightBonds=[],highlightAtomColors=atom_colors)
	drawer.FinishDrawing()
	svg = drawer.GetDrawingText()
	svg = svg.replace('svg:','')
	display(SVG(svg))
	svg.save('example.svg')


smiles='CC(C)(CC=C)O'
weights=[-0.7706536253293356, -0.8106458981831869, -0.8747469782829285, 
-0.8747469782829285, -0.8632091482480367, -0.7221730351448059, -0.8728522658348083]
weights=np.array(weights)
scaler = MinMaxScaler()
norm = scaler.fit_transform(weights.reshape(-1,1))
weights=torch.Tensor(norm)


draw(smiles,weights)