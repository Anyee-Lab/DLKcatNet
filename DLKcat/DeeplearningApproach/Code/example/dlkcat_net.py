
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

fingerprint_path = 'DeeplearningApproach/Data/input/fingerprint_dict.pickle'
atom_path = 'DeeplearningApproach/Data/input/atom_dict.pickle'
bond_path = 'DeeplearningApproach/Data/input/bond_dict.pickle'
edge_path = 'DeeplearningApproach/Data/input/edge_dict.pickle'
word_path = 'DeeplearningApproach/Data/input/sequence_dict.pickle'
model_path = 'DeeplearningApproach/Results/output/1.5--decay_interval10--weight_decay1e-6--iteration50'

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

# Utility function to load dictionaries
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

class DLKcatPrediction:
    def __init__(self, path):
        self.fingerprint_dict = model.load_pickle(os.path.join(path, fingerprint_path))
        self.atom_dict = model.load_pickle(os.path.join(path, atom_path))
        self.bond_dict = model.load_pickle(os.path.join(path, bond_path))
        self.word_dict = model.load_pickle(os.path.join(path, word_path))
        self.edge_dict = model.load_pickle(os.path.join(path, edge_path))

        self.n_fingerprint = len(self.fingerprint_dict)
        self.n_word = len(self.word_dict)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("[DLKcat]: Device is GPU")
        else:
            self.device = torch.device('cpu')
            print("[DLKcat]: Device is CPU")

        self.Kcat_model = model.KcatPrediction(self.device, self.n_fingerprint, self.n_word, 2*dim, layer_gnn, window, layer_cnn, layer_output).to(self.device)
        self.Kcat_model.load_state_dict(torch.load(os.path.join(path,model_path), map_location=self.device))

    def _predict(self, data):
        with torch.no_grad(): # 添加 no_grad 提高推理效率
            predicted_value = self.Kcat_model.forward(data)
        return predicted_value

    def device_set(self, device):
        self.device = device

    def predict_for_one(self, name, smiles, sequence):
        if smiles == 'None' :
            smiles = get_smiles(name)

        # try :
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol,self.atom_dict)
        i_jbond_dict = create_ijbonddict(mol,self.bond_dict)
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius, self.fingerprint_dict, self.edge_dict)
        adjacency = create_adjacency(mol)
        words = split_sequence(sequence,ngram,self.word_dict)

        fingerprints = torch.LongTensor(fingerprints).to(self.device)
        adjacency = torch.FloatTensor(adjacency).to(self.device)
        words = torch.LongTensor(words).to(self.device)

        inputs = [fingerprints, adjacency, words]

        prediction = self._predict(inputs)
        Kcat_log_value = prediction.item()
        Kcat_value = '%.4f' %math.pow(2,Kcat_log_value)

        return Kcat_value

    def predict_for_input(self, name):
        with open(name, 'r') as infile :
            lines = infile.readlines()

        with open('./output.tsv', 'w') as outfile :
            items = ['Substrate Name', 'Substrate SMILES', 'Protein Sequence', 'Kcat value (1/s)']
            outfile.write('\t'.join(items)+'\n')

            for line in lines[1:] :
                line_item = list()
                data = line.strip().split('\t')

                name = data[0]
                smiles = data[1]
                sequence = data[2]
                if smiles and smiles != 'None' :
                    smiles = data[1]
                else :
                    smiles = get_smiles(name)

                try :
                    if "." not in smiles :
                        Kcat_value = self.predict_for_one(name, smiles, sequence)
                        line_item = [name,smiles,sequence,Kcat_value]
                        outfile.write('\t'.join(line_item)+'\n')
                except :
                    Kcat_value = 'None'
                    line_item = [name,smiles,sequence,Kcat_value]
                    outfile.write('\t'.join(line_item)+'\n')

        print('[DLKcat]: Prediction is over!')








