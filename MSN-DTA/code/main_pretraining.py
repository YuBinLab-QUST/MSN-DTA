import numpy as np
import rdkit
import rdkit.Chem as Chem
import networkx as nx
from pre_vocab import WordVocab
import pandas as pd
import torch.nn as nn
from prepare_data import DTADataset,Pre_DTADataset
from utils import *
from MSGIN import *
from torch_geometric.loader import DataLoader


class MSNDTA_pretraining(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, hidden_dim, dropout_rate,
                 alpha, n_heads, bilstm_layers=2, protein_vocab=26, smile_vocab=45, theta=0.5):
        super(MSNDTA_pretraining, self).__init__()
        self.is_bidirectional = True

        self.theta = theta
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.bilstm_layers = bilstm_layers
        self.n_heads = n_heads


        self.smiles_vocab = smile_vocab
        self.smiles_embed = nn.Embedding(smile_vocab +1, 256, padding_idx=0)

        self.is_bidirectional = True
        self.smiles_input_fc = nn.Linear(256, lstm_dim)
        self.smiles_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                   bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln1 = torch.nn.LayerNorm(lstm_dim * 2)
        self.out_attentions3 = LinkAttention(hidden_dim, n_heads)


        self.CNNProtein = Mscnn()
        self.protein_vocab = protein_vocab
        self.protein_embed = nn.Embedding(protein_vocab + 1, embedding_dim, padding_idx=0)
        self.is_bidirectional = True
        self.protein_input_fc= nn.Linear(embedding_dim, lstm_dim)
        self.protein_lstm = nn.LSTM(lstm_dim, lstm_dim, self.bilstm_layers, batch_first=True,
                                    bidirectional=self.is_bidirectional, dropout=dropout_rate)
        self.ln2 = torch.nn.LayerNorm(lstm_dim * 2)
        self.protein_head_fc = nn.Linear(lstm_dim * n_heads, lstm_dim)
        self.protein_out_fc = nn.Linear(2 * lstm_dim, hidden_dim)
        self.out_attentions2 = LinkAttention(hidden_dim, n_heads)


        self.out_attentions = LinkAttention(hidden_dim, n_heads)
        self.out_fc1 = nn.Linear(hidden_dim * 3, 256 * 8)
        self.out_fc2 = nn.Linear(256 * 8, hidden_dim * 2)
        self.out_fc3 = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(lstm_dim * 2)

    def forward(self, data, reset=False):
        protein, smiles = data[1].cuda(), data[0].cuda()
        smiles_lengths = data[-2].cuda()
        protein_lengths = data[-1].cuda()
        batchsize = len(protein)

        smiles = self.smiles_embed(smiles)

        smiles = self.smiles_input_fc(smiles)
        smiles, _ = self.smiles_lstm(smiles)
        smiles = self.ln1(smiles)

        protein_emb = self.protein_embed(protein)
        protein_cnn = self.CNNProtein(protein_emb)
        protein = self.protein_input_fc(protein_emb)
        protein, _ = self.protein_lstm(protein)

        protein = self.ln2(protein)
        if reset:
            return smiles, protein

        smiles_mask = self.generate_mask(smiles, smiles_lengths, self.n_heads)  # batch * head* seq len
        smiles_out, smile_attn = self.out_attentions3(smiles, smiles_mask)  # batch * lstm_dim*2

        protein_mask = self.generate_mask(protein, protein_lengths, self.n_heads)  # batch * head * tar_len
        protein_out, prot_attn = self.out_attentions2(protein, protein_mask)  # B * (lstm_dim *2)


        out_cat = torch.cat((smiles, protein), dim=1)  # batch * head * lstm_dim *2
        out_masks = torch.cat((smiles_mask, protein_mask), dim=2)  # batch * tar_len+seq_len * (lstm_dim *2)
        out_cat, out_attn = self.out_attentions(out_cat, out_masks)
        smiledpoutput = torch.zeros(128,128,device=smiles.device)
        out = torch.cat([smiles_out, protein_out, (out_cat+smiledpoutput)], dim=-1)
        out = self.dropout(self.relu(self.out_fc1(out)))
        out = self.dropout(self.relu(self.out_fc2(out)))
        out = self.out_fc3(out).squeeze()

        del smiles_out, protein_out
        return out + protein_cnn

    def generate_mask(self, adj, adj_sizes, n_heads):
        out = torch.ones(adj.shape[0], adj.shape[1])
        max_size = adj.shape[1]
        if isinstance(adj_sizes, int ):
            out[0, adj_sizes:max_size] = 0
        else:
            for e_id, drug_len in enumerate(adj_sizes):
                out[e_id, drug_len:max_size] = 0
        out = out.unsqueeze(1).expand(-1, n_heads, -1)
        return out.cuda(device=adj.device)


################################################################

CUDA = '4'
dataset_name = 'davis'
seed = 0
reset_epoch = 40
drug_vocab = WordVocab.load_vocab('../Vocab/smiles_vocab.pkl')
target_vocab = WordVocab.load_vocab('../Vocab/protein_vocab.pkl')

tar_len = 2600
seq_len = 536
load_model_path = None

LR = 1e-3
NUM_EPOCHS = 500
model_file_name = 'output_file_name'

embedding_dim = 128
lstm_dim = 64
hidden_dim = 128
dropout_rate = 0.1
alpha = 0.2
n_heads = 8

#############################
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA

device = torch.device('cuda:0')

seed_torch(seed)



def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size,c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    edge_index = np.array(edge_index)
    return c_size, edge_index



df = pd.read_csv('../data/' + dataset_name + '.csv')
smiles = set(df['compound_iso_smiles'])
target = set(df['target_key'])

target_seq = {}
for i in range(len(df)):
    target_seq[df.loc[i, 'target_key']] = df.loc[i, 'target_sequence']
smiles_graph = {}
for sm in smiles:
    _, graph = smiles_to_graph(sm)
    smiles_graph[sm] = graph

target_emb = {}
target_len = {}
for k in target_seq:
    seq = target_seq[k]
    content = []
    flag = 0
    for i in range(len(seq)):
        if flag >= len(seq):
            break
        if (flag + 1 < len(seq)):
            if target_vocab.stoi.__contains__(seq[flag:flag + 2]):
                content.append(target_vocab.stoi.get(seq[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(target_vocab.stoi.get(seq[flag], target_vocab.unk_index))
        flag = flag + 1

    if len(content) > tar_len:
        content = content[:tar_len]

    X = [target_vocab.sos_index] + content + [target_vocab.eos_index]
    target_len[seq] = len(content)
    if tar_len > len(X):
        padding = [target_vocab.pad_index] * (tar_len - len(X))
        X.extend(padding)
    target_emb[seq] = torch.tensor(X)

smiles_idx = {}
smiles_emb = {}
smiles_len = {}

for sm in smiles:

    content = []
    flag = 0
    for i in range(len(sm)):
        if flag >= len(sm):
            break
        if (flag + 1 < len(sm)):
            if drug_vocab.stoi.__contains__(sm[flag:flag + 2]):
                content.append(drug_vocab.stoi.get(sm[flag:flag + 2]))
                flag = flag + 2
                continue
        content.append(drug_vocab.stoi.get(sm[flag], drug_vocab.unk_index))
        flag= flag + 1

    if len(content) > seq_len:
        content = content[:seq_len]

    X = [drug_vocab.sos_index] + content + [drug_vocab.eos_index]
    smiles_len[sm] = len(content)
    if seq_len > len(X):
        padding = [drug_vocab.pad_index] * (seq_len - len(X))
        X.extend(padding)

    smiles_emb[sm] = torch.tensor(X)

    if not smiles_idx.__contains__(sm):
        tem = []
        for i, c in enumerate(X):
            if atom_dict.__contains__(c):
                tem.append(i)
        smiles_idx[sm] = tem



print('built model...')

model = MSNDTA_pretraining(embedding_dim = embedding_dim, lstm_dim = lstm_dim, hidden_dim = hidden_dim,
                             dropout_rate = dropout_rate, alpha = alpha, n_heads = n_heads).to(device)

if load_model_path is not None:
    print(load_model_path)
    save_model = torch.load(load_model_path)
    model_dict = model.state_dict()
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
batch_size = 128

print('create dataset...')
dataset = Pre_DTADataset(path = '../Data/' + dataset_name + '.csv', drug_vocab = drug_vocab, target_vocab = target_vocab,
                         smiles_emb = smiles_emb, target_emb = target_emb, smiles_len = smiles_len, target_len = target_len)
test_size = (int)(len(dataset) * 0.1)

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset = dataset,
    lengths = [len(dataset) - (test_size * 2), test_size * 2],
    generator = torch.Generator().manual_seed(0)
)

val_dataset, test_dataset = torch.utils.data.random_split(
    dataset = test_dataset,
    lengths = [test_size, test_size],
    generator = torch.Generator().manual_seed(0)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('Train size is:', len(train_loader))
print('Test size is:', len(val_loader))
print('Test size is:', len(test_loader))

device = torch.device('cuda:0')

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=5e-4, last_epoch=-1)

best_mse = 1000
best_test_mse = 1000
best_epoch = -1
best_test_epoch = -1

file_MSEs = ''

for epoch in range(NUM_EPOCHS):
    print("No {} epoch".format(epoch))
    pre_train(model, train_loader, optimizer, epoch)
    G, P = pre_predicting(model, val_loader)
    vall = get_mse(G, P)
    if vall < best_mse:
        best_mse = vall
        best_epoch = epoch + 1
        if model_file_name is not None:
            torch. save(model.state_dict(), model_file_name)
        print('mse improved at epoch ', best_epoch, '; best mse', best_mse)
    else:
        print('current mse', vall, 'No improvement since epoch', best_epoch, '; best mse', best_mse)
    MSEs = [epoch, vall, best_mse]
    save_mse(MSEs,file_MSEs)
    schedule.step()


