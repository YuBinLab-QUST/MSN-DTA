import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch.nn.modules.batchnorm import _BatchNorm
import torch_geometric.nn as gnn
from torch import Tensor
from collections import OrderedDict
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
'''
MSN-DTA
'''

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=128, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        dim = 128
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)  # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x.cuda(), data.edge_index.cuda(), data.batch.cuda()
        # target = data.target
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # concat
        xc = x
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

class sequence_process(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num, lstm_dim=64, dropout_rate=0.1, bilstm_layers=1, n_heads=8,
                 sm_feature=22):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                MultiCNN(block_idx + 1, embedding_num, 96, 3)
            )
        self.linear = nn.Linear(block_num * 96 + 192, 96)
        self.bilstm_layers = bilstm_layers
        self.protein_input_fc = nn.Linear(128, lstm_dim)
        self.protein_lstm = nn.LSTM(lstm_dim, 96, self.bilstm_layers, batch_first=True,
                                    bidirectional=True, dropout=dropout_rate)
        self.attention1 = LinkAttention(192, n_heads)

        self.smile_input_fc = nn.Linear(sm_feature, lstm_dim)
        self.smile_lstm = nn.LSTM(lstm_dim, 96, self.bilstm_layers, batch_first=True,
                                    bidirectional=True, dropout=dropout_rate)
        self.out_node_sm = nn.Linear(192, 22)
        self.attention2 = SMAttention(192, n_heads)
        self.linear2 = nn.Linear(192, 96)


    def forward(self, x, data, reset=False, pretrain=False):
        x = self.embed(x).permute(0, 2, 1)

        protein = x.permute(0, 2, 1)
        protein = self.protein_input_fc(protein)
        protein, _ = self.protein_lstm(protein)  #batchsize * tar_length * embedding_size
        protein_out, pro_attn = self.attention1(protein) #  batchsize * embedding

        if pretrain:
            y = data.x
            smile = self.smile_input_fc(y)
            smile = smile.unsqueeze(0)
            smile,_ = self.smile_lstm(smile) #smile: batchsize * ligandlen * embedding
            smile = smile.squeeze(0)
            smile = gnn.global_mean_pool(smile, data.batch)
            smile = self.linear2(smile)

            feats = [block(x) for block in self.block_list]
            x = torch.cat(feats, -1)
            x = torch.cat((x, protein_out), dim=-1)
            x = self.linear(x)

            return x, smile

        if reset:
            y = data.x
            smile = self.smile_input_fc(y)
            smile = smile.unsqueeze(0)
            smile, _ = self.smile_lstm(smile)
            smile = smile.squeeze(0)

            node_update = smile
            node_update = self.out_node_sm(node_update)
            data.x = node_update
            reset = False

            feats = [block(x) for block in self.block_list]
            x = torch.cat(feats, -1)
            x = torch.cat((x, protein_out), dim=-1)
            x = self.linear(x)

            return x

        else:
            feats = [block(x) for block in self.block_list]
            x = torch.cat(feats, -1)
            x = torch.cat((x, protein_out), dim=-1)
            x = self.linear(x)

            return x
        # protein = protein.permute(0, 2, 1)
        # Amaxpool = nn.AdaptiveMaxPool1d(1)
        # protein = Amaxpool(protein)
        # protein = protein.permute(0, 2, 1)
        # protein = protein.squeeze(1)



class Conv1dReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.inc(x)

class MultiCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.mcnn = nn.Sequential(OrderedDict([('cnn_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size,
                                                                          stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self. mcnn.add_module('cnn_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size,
                                                                              stride=stride, padding=padding))

        self.mcnn.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))

    def forward(self, x):
        x = self.mcnn(x).squeeze(-1)
        return x



class NodeLevelBatchNorm(_BatchNorm):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NodeLevelBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def _check_input_dim(self, input):
        if input.dim() != 2:
            raise ValueError('expected 2D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        return torch.functional.F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={eps}, ' \
               'affine={affine}'.format(**self.__dict__)


class GraphConvBn(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = gnn.GraphConv(in_channels, out_channels)
        self.norm = NodeLevelBatchNorm(out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = F.relu(self.norm(self.conv(x, edge_index)))

        return data


class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        self.conv1 = GraphConvBn(num_input_features, int(growth_rate * bn_size))
        self.conv2 = GraphConvBn(int(growth_rate * bn_size), growth_rate)

    def bn_function(self, data):
        concated_features = torch.cat(data.x, 1)
        data.x = concated_features

        data = self.conv1(data)

        return data

    def forward(self, data):
        if isinstance(data.x, Tensor):
            data.x = [data.x]

        data = self.bn_function(data)
        data = self.conv2(data)

        return data


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, growth_rate=32, bn_size=4):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size)
            self.add_module('layer%d' % (i + 1), layer)

    def forward(self, data):
        features = [data.x]
        for name, layer in self.items():
            data = layer(data)
            features.append(data.x)
            data.x = features

        data.x = torch.cat(data.x, 1)

        return data


class GraphDenseNet(nn.Module):
    def __init__(self, num_input_features, out_dim, growth_rate=32, block_config=(3, 3, 3, 3), bn_sizes=[2, 3, 4, 4]):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([('conv0', GraphConvBn(num_input_features, 32))]))
        num_input_features = 32

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers, num_input_features, growth_rate=growth_rate, bn_size=bn_sizes[i]
            )
            self.features.add_module('block%d' % (i + 1), block)
            num_input_features += int(num_layers * growth_rate)

            trans = GraphConvBn(num_input_features, num_input_features // 2)
            self.features.add_module("transition%d" % (i + 1), trans)
            num_input_features = num_input_features // 2

        self.classifer = nn.Linear(num_input_features, out_dim)

    def forward(self, data):
        mgdata = self.features(data)
        x = gnn.global_mean_pool(mgdata.x, mgdata.batch)
        x = self.classifer(x)

        return x


class MSN_DTA(nn.Module):
    def __init__(self, block_num, sm_feature, vocab_protein_size, embedding_size=128, filter_num=32, out_dim=1,
                 n_heads=8, blpretrain=False):
        super().__init__()
        self.protein_encoder = sequence_process(block_num, vocab_protein_size, embedding_size, sm_feature=sm_feature)
        self.ligand_encoder = GraphDenseNet(num_input_features=22, out_dim=filter_num * 3, block_config=[8, 8, 8],
                                            bn_sizes=[2, 2, 2])
        self.n_heads = n_heads
        self.blpretrain = blpretrain

        self.classifier = nn.Sequential(
            nn.Linear(filter_num * 3 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, out_dim)
        )

    def forward(self, data, epochs, reset_epochs):
        target = data.target

        if self.blpretrain:
            protein_x, smile = self.protein_encoder(target, data, pretrain=True)
            x = torch.cat([protein_x, smile], dim=-1)
            x = self.classifier(x)
            return x

        elif epochs % reset_epochs == 0:
            protein_x = self.protein_encoder(target, data, reset=True)
            ligand_x = self.ligand_encoder(data)
            x = torch.cat([protein_x, ligand_x], dim=-1)
            x = self.classifier(x)
            return x

        else:
            protein_x = self.protein_encoder(target, data)
            ligand_x = self.ligand_encoder(data)
            x = torch.cat([protein_x, ligand_x], dim=-1)
            x = self.classifier(x)
            return x




class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x).transpose(1, 2)
        value = x
        a = self.softmax(query)

        out = torch.matmul(a, value)
        out = torch.sum(out, dim=1).squeeze()
        return out, a

class SMAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(SMAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x).transpose(1, 2)
        value = x
        a = self.softmax(query)

        out = torch.matmul(a, value)
        return out, a

