#!/usr/bin/python
# coding: utf-8

# Editor: ZHENG YI
# Email: Anyee1277@outlook.com

import os
import sys
import math
import model
import torch
import requests
import pickle
import numpy as np
from rdkit import Chem
from collections import defaultdict

fingerprint_path = 'DLKcat/DeeplearningApproach/Data/input/fingerprint_dict.pickle'
atom_path = 'DLKcat/DeeplearningApproach/Data/input/atom_dict.pickle'
bond_path = 'DLKcat/DeeplearningApproach/Data/input/bond_dict.pickle'
edge_path = 'DLKcat/DeeplearningApproach/Data/input/edge_dict.pickle'
word_path = 'DLKcat/DeeplearningApproach/Data/input/sequence_dict.pickle'
model_path = 'DLKcat/DeeplearningApproach/Results/output/1.5--decay_interval10--weight_decay1e-6--iteration50'

def split_sequence(sequence, ngram, word_dict):
    sequence = '-' + sequence + '='

    words = list()
    for i in range(len(sequence)-ngram+1) :
        try :
            words.append(word_dict[sequence[i:i+ngram]])
        except :
            word_dict[sequence[i:i+ngram]] = 0
            words.append(word_dict[sequence[i:i+ngram]])

    return np.array(words)
    # return word_dict

def create_atoms(mol,atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol,bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict):

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                try :
                    fingerprints.append(fingerprint_dict[fingerprint])
                except :
                    fingerprint_dict[fingerprint] = 0
                    fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    try :
                        edge = edge_dict[(both_side, edge)]
                    except :
                        edge_dict[(both_side, edge)] = 0
                        edge = edge_dict[(both_side, edge)]

                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

class Predictor(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, data):
        # 确保输入张量在正确设备
        data = [d.to(self.device) if isinstance(d, torch.Tensor) else d for d in data]
        with torch.no_grad():  # 添加 no_grad 提高推理效率
            predicted_value = self.model.forward(data)
        return predicted_value

def get_smiles(name):
    try :
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT' % name
        req = requests.get(url)
        if req.status_code != 200:
            smiles = None
        else:
            smiles = req.content.splitlines()[0].decode()
    except :
        smiles = None
    return smiles

def predict_for_one(name, smiles, sequence, path):

    fingerprint_dict = model.load_pickle(os.path.join(path, fingerprint_path))
    atom_dict = model.load_pickle(os.path.join(path, atom_path))
    bond_dict = model.load_pickle(os.path.join(path, bond_path))
    word_dict = model.load_pickle(os.path.join(path, word_path))
    edge_dict = model.load_pickle(os.path.join(path, edge_path))
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    radius=2
    ngram=3

    dim=10
    layer_gnn=3
    side=5
    window=11
    layer_cnn=3
    layer_output=3
    lr=1e-3
    lr_decay=0.5
    decay_interval=10
    weight_decay=1e-6
    iteration=100

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    Kcat_model = model.KcatPrediction(device, n_fingerprint, n_word, 2*dim, layer_gnn, window, layer_cnn, layer_output).to(device)
    Kcat_model.load_state_dict(torch.load(os.path.join(path,model_path), map_location=device))
    predictor = Predictor(Kcat_model, device)

    print("predict is running")

    if smiles == 'None' :
        smiles = get_smiles(name)

    # try :
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    atoms = create_atoms(mol,atom_dict)
    i_jbond_dict = create_ijbonddict(mol,bond_dict)
    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius, fingerprint_dict, edge_dict)
    adjacency = create_adjacency(mol)
    words = split_sequence(sequence,ngram,word_dict)

    fingerprints = torch.LongTensor(fingerprints)
    adjacency = torch.FloatTensor(adjacency)
    words = torch.LongTensor(words)

    inputs = [fingerprints, adjacency, words]

    prediction = predictor.predict(inputs)
    Kcat_log_value = prediction.item()
    Kcat_value = '%.4f' %math.pow(2,Kcat_log_value)
    # except :
    #     Kcat_value = 'None'
    #     print("-1")


    return Kcat_value



if __name__ == '__main__' :
    predict_for_one()

