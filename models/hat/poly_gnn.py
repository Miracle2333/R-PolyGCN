import torch
import torch.nn as nn
import Utils.utils as utils
import torch.nn.functional as F
from Models.Encoder.resnet_GCN_skip import SkipResnet50
from first_annotation import FirstAnnotation
from GCN import GCN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolyGNN(nn.Module):
    def __init__(self,
                 state_dim=256,
                 n_adj=6,
                 cnn_feature_grids=None,
                 coarse_to_fine_steps=0,
                 get_point_annotation=False
                 ):

        super(PolyGNN, self).__init__()

        self.state_dim = state_dim
        self.n_adj = n_adj
        self.cnn_feature_grids = cnn_feature_grids
        self.coarse_to_fine_steps = coarse_to_fine_steps
        self.get_point_annotation = get_point_annotation

        print 'Building GNN Encoder'


        if get_point_annotation:
            nInputChannels = 4
        else:
            nInputChannels = 3


        self.encoder = SkipResnet50(nInputChannels=nInputChannels,
                                    classifier='psp',
                                    cnn_feature_grids=self.cnn_feature_grids)


        self.grid_size = self.encoder.feat_size

        self.psp_feature = [self.cnn_feature_grids[-1]]

        #The number of GCN needed
        if self.coarse_to_fine_steps > 0:
            for step in range(self.coarse_to_fine_steps):
                if step == 0:
                    self.gnn = nn.ModuleList(
                        [GCN(state_dim=self.state_dim, feature_dim=self.encoder.final_dim + 2).to(device)])
                else:
                    self.gnn.append(GCN(state_dim=self.state_dim, feature_dim=self.encoder.final_dim + 2).to(device))
        else:

            self.gnn = GCN(state_dim=self.state_dim, feature_dim=self.encoder.final_dim + 2)

        # vertex and edge prediction, selection of the first vertex
        self.first_annotation = FirstAnnotation(28, 512, 16)
        
        #Initialize the weight for different layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                # m.weight.data.normal_(0.0, 0.00002)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) 
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x, init_polys):
        """
        pred_polys: in scale [0,1]
        """
        image_size = x.size(2), x.size(3)
        out_dict = {}

        conv_layers = self.encoder.forward(x, return_final_res_feature=True)
        conv_layers, psp_out = conv_layers[:-1], conv_layers[-1]

        edge_logits, vertex_logits, logprob, _ = self.first_annotation.forward(psp_out)
        out_dict['edge_logits'] = edge_logits
        out_dict['vertex_logits'] = vertex_logits

        del(conv_layers)

        edge_logits = edge_logits.view(
            (-1, self.first_annotation.grid_size, self.first_annotation.grid_size)).unsqueeze(1)
        vertex_logits = vertex_logits.view(
            (-1, self.first_annotation.grid_size, self.first_annotation.grid_size)).unsqueeze(1)

        feature_with_edges = torch.cat([psp_out, edge_logits, vertex_logits], 1)

        out_dict['feature_with_edges'] = feature_with_edges
        conv_layers = [self.encoder.edge_annotation_cnn(feature_with_edges)]


        out_dict['pred_polys'] = []

        for i in range(self.coarse_to_fine_steps):
            if i == 0:
                component = utils.prepare_gcn_component(init_polys.numpy(),
                                                        self.psp_feature,
                                                        init_polys.size()[1],
                                                        n_adj=self.n_adj)
                init_polys = init_polys.to(device)
                adjacent = component['adj_matrix'].to(device)
                init_poly_idx = component['feature_indexs'].to(device)

                cnn_feature = self.encoder.sampling(init_poly_idx,
                                                        conv_layers)
                input_feature = torch.cat((cnn_feature, init_polys), 2)

            else:
                init_polys = gcn_pred_poly
                cnn_feature = self.interpolated_sum(conv_layers, init_polys, self.psp_feature)
                input_feature = torch.cat((cnn_feature, init_polys), 2)

            gcn_pred = self.gnn[i].forward(input_feature, adjacent)
            gcn_pred_poly = init_polys.to(device) + gcn_pred

            out_dict['pred_polys'].append(gcn_pred_poly)
        out_dict['adjacent'] = adjacent

        return out_dict


    def interpolated_sum(self, cnns, coords, grids):

        X = coords[:,:,0]
        Y = coords[:,:,1]

        cnn_outs = []
        for i in range(len(grids)):
            grid = grids[i]

            Xs = X * grid
            X0 = torch.floor(Xs)
            X1 = X0 + 1

            Ys = Y * grid
            Y0 = torch.floor(Ys)
            Y1 = Y0 + 1

            w_00 = (X1 - Xs) * (Y1 - Ys)
            w_01 = (X1 - Xs) * (Ys - Y0)
            w_10 = (Xs - X0) * (Y1 - Ys)
            w_11 = (Xs - X0) * (Ys - Y0)

            X0 = torch.clamp(X0, 0, grid-1)
            X1 = torch.clamp(X1, 0, grid-1)
            Y0 = torch.clamp(Y0, 0, grid-1)
            Y1 = torch.clamp(Y1, 0, grid-1)

            N1_id = X0 + Y0 * grid
            N2_id = X0 + Y1 * grid
            N3_id = X1 + Y0 * grid
            N4_id = X1 + Y1 * grid

            M_00 = utils.gather_feature(N1_id, cnns[i])
            M_01 = utils.gather_feature(N2_id, cnns[i])
            M_10 = utils.gather_feature(N3_id, cnns[i])
            M_11 = utils.gather_feature(N4_id, cnns[i])
            cnn_out = w_00.unsqueeze(2) * M_00 + \
                      w_01.unsqueeze(2) * M_01 + \
                      w_10.unsqueeze(2) * M_10 + \
                      w_11.unsqueeze(2) * M_11

            cnn_outs.append(cnn_out)
        concat_features = torch.cat(cnn_outs, dim=2)
        return concat_features



    def reload(self, path, strict=False):
        print "Reloading full model from: ", path
        self.load_state_dict(torch.load(path)['state_dict'],
            strict=strict)