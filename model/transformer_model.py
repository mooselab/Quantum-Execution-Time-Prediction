import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.nn.norm import LayerNorm


class Simple_Model(torch.nn.Module):
    def __init__(self, args, length_of_x=178, length_of_gf=41):
        super().__init__()
        self.args = args
        self.length_of_x = length_of_x # graph features
        self.length_of_gf = length_of_gf

        if self.args.use_graph_features and self.args.use_global_features:
            self.gf_linear1 = torch.nn.Linear(self.length_of_gf, 64)
            self.gf_linear2 = torch.nn.Linear(64, 64)
            self.linear1 = torch.nn.Linear(self.length_of_x+64, 512)
        elif self.args.use_graph_features and not self.args.use_global_features:
            self.linear1 = torch.nn.Linear(self.length_of_x, 512)
        else:
            self.gf_linear1 = torch.nn.Linear(self.length_of_gf, 64)
            self.gf_linear2 = torch.nn.Linear(64, 64)
            self.linear1 = torch.nn.Linear(64, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 128)
        self.linear4 = torch.nn.Linear(128, 1)
        if self.args.use_graph_features:
            self.mask = torch.full((self.length_of_x,), True)
            if not self.args.use_gate_type:
                self.mask[0:46] = False
            if not self.args.use_qubit_index:
                self.mask[46:173] = False
            if not self.args.use_T1T2:
                self.mask[173:177] = False
            if not self.args.use_gate_index:
                self.mask[177] = False
            lenth_of_mask = 0
            for i in self.mask:
                if i:
                    lenth_of_mask += 1
            setattr(self, f"conv{0}", TransformerConv(lenth_of_mask, self.length_of_x))
            for i in range(1, self.args.num_layers):
                setattr(self, f"conv{i}", TransformerConv(self.length_of_x, self.length_of_x))

    def forward(self, data):
        x, edge_index, gf = data.x, data.edge_index, data.global_features # x is graph features
        x = x.to(torch.float32)
        gf = gf.to(torch.float32)
        if self.args.use_graph_features:
            x = x[:, self.mask]
            for i in range(self.args.num_layers):
                x = getattr(self, f"conv{i}")(x, edge_index)
                x = F.relu(x)
            x = global_mean_pool(x, data.batch)
        if self.args.use_global_features:
            gf = self.gf_linear1(gf)
            gf = F.relu(gf)
            gf = self.gf_linear2(gf)
            gf = F.relu(gf)
            if self.args.use_graph_features:
                x = torch.cat([x, gf], dim=1)
            else:
                x = gf
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        return x.squeeze()
