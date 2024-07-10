import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
import pickle

class GeneralizedRelationalConv(MessagePassing):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)

        if dependent:
            # obtain relation embeddings as a projection of the query relation
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            # relation embeddings as an independent embedding matrix per each layer
            self.relation = nn.Embedding(num_relation, input_dim)

    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None):
        batch_size = len(query)


        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            # layer-specific relation features as a special embedding matrix unique to each layer
            relation = self.relation.weight.expand(batch_size, -1, -1)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)
        if torch.isnan(edge_weight).any().item():
            raise ValueError("edge_weight has nan")
        if torch.isnan(relation).any().item():
            raise ValueError("relation has nan")
        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight)
        if torch.isnan(output).any().item():
            raise ValueError("output has nan")
        # output=[]
        # for i in range(batch_size):
        #     o = self.propagate(input=input[i,None], relation=relation[i,None], boundary=boundary[i,None], edge_index=edge_index,
        #                             edge_type=edge_type, size=size, edge_weight=edge_weight)
        #     output.append(o)
        # output=torch.cat(output,dim=0)
        return output

    def propagate(self, edge_index, size=None, **kwargs):
        return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)
        # if kwargs["edge_weight"].requires_grad or self.message_func == "rotate":
        #     # the rspmm cuda kernel only works for TransE and DistMult message functions
        #     # otherwise we invoke separate message & aggregate functions
        #     return super(GeneralizedRelationalConv, self).propagate(edge_index, size, **kwargs)
        #
        # for hook in self._propagate_forward_pre_hooks.values():
        #     res = hook(self, (edge_index, size, kwargs))
        #     if res is not None:
        #         edge_index, size, kwargs = res
        #
        # size = self.__check_input__(edge_index, size)
        # coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
        #                              size, kwargs)
        #
        # msg_aggr_kwargs = self.inspector.distribute("message_and_aggregate", coll_dict)
        # for hook in self._message_and_aggregate_forward_pre_hooks.values():
        #     res = hook(self, (edge_index, msg_aggr_kwargs))
        #     if res is not None:
        #         edge_index, msg_aggr_kwargs = res
        # out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
        # for hook in self._message_and_aggregate_forward_hooks.values():
        #     res = hook(self, (edge_index, msg_aggr_kwargs), out)
        #     if res is not None:
        #         out = res
        #
        # update_kwargs = self.inspector.distribute("update", coll_dict)
        # out = self.update(out, **update_kwargs)
        #
        # for hook in self._propagate_forward_hooks.values():
        #     res = hook(self, (edge_index, size, kwargs), out)
        #     if res is not None:
        #         out = res
        #
        # return out

    def message(self, input_j, relation, boundary, edge_type):
        relation_j = relation.index_select(self.node_dim, edge_type)
        if torch.isnan(relation_j).any().item():
            raise ValueError("relation_j has nan")
        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        # augment messages with the boundary condition
        message = torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)
        if torch.isnan(message).any().item():
            raise ValueError("message has nan")
        return message

    def aggregate(self, input, edge_weight, index, dim_size):
        # augment aggregation index with self-loops for the boundary condition
        index = torch.cat([index, torch.arange(dim_size, device=input.device)]) # (num_edges + num_nodes,)
        if torch.isnan(index).any().item():
            raise ValueError("index has nan")
        edge_weight = torch.cat([edge_weight, torch.ones(dim_size, device=input.device)])
        if torch.isnan(edge_weight).any().item():
            raise ValueError("edge_weight has nan")
        shape = [1] * input.ndim
        shape[self.node_dim] = -1
        edge_weight = edge_weight.view(shape)

        if self.aggregate_func == "pna":
            mean = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            if torch.isnan(mean).any().item():
                raise ValueError("mean has nan")
            sq_mean = scatter(input ** 2 * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")
            if torch.isnan(sq_mean).any().item():
                raise ValueError("sq_mean has nan")
            max = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="max")
            if torch.isnan(max).any().item():
                raise ValueError("max has nan")
            min = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size, reduce="min")
            if torch.isnan(min).any().item():
                raise ValueError("min has nan")
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            if torch.isnan(std).any().item():
                raise ValueError("std has nan")
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            if torch.isnan(features).any().item():
                raise ValueError("features has nan")
            degree_out = degree(index, dim_size).unsqueeze(0).unsqueeze(-1).clamp(min=1)
            if torch.isnan(degree_out).any().item():
                raise ValueError("degree_out has nan")
            scale = degree_out.log()
            scale = scale / scale.mean() if scale.mean() > 0 else scale
            if torch.isnan(scale).any().item():
                raise ValueError("scale has nan")
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            if torch.isnan(scales).any().item():
                raise ValueError("scales has nan")
            output = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            output = scatter(input * edge_weight, index, dim=self.node_dim, dim_size=dim_size,
                             reduce=self.aggregate_func)
        if torch.isnan(output).any().item():
            raise ValueError("output has nan")

        return output

    # def message_and_aggregate(self, edge_index, input, relation, boundary, edge_type, edge_weight, index, dim_size):
    #     # fused computation of message and aggregate steps with the custom rspmm cuda kernel
    #     # speed up computation by several times
    #     # reduce memory complexity from O(|E|d) to O(|V|d), so we can apply it to larger graphs
    #     from .rspmm import generalized_rspmm
    #
    #     batch_size, num_node = input.shape[:2]
    #     input = input.transpose(0, 1).flatten(1)
    #     relation = relation.transpose(0, 1).flatten(1)
    #     boundary = boundary.transpose(0, 1).flatten(1)
    #     degree_out = degree(index, dim_size).unsqueeze(-1) + 1
    #
    #     if self.message_func in self.message2mul:
    #         mul = self.message2mul[self.message_func]
    #     else:
    #         raise ValueError("Unknown message function `%s`" % self.message_func)
    #     if self.aggregate_func == "sum":
    #         update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
    #         update = update + boundary
    #     elif self.aggregate_func == "mean":
    #         update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
    #         update = (update + boundary) / degree_out
    #     elif self.aggregate_func == "max":
    #         update = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
    #         update = torch.max(update, boundary)
    #     elif self.aggregate_func == "pna":
    #         # we use PNA with 4 aggregators (mean / max / min / std)
    #         # and 3 scalars (identity / log degree / reciprocal of log degree)
    #         sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="add", mul=mul)
    #         sq_sum = generalized_rspmm(edge_index, edge_type, edge_weight, relation ** 2, input ** 2, sum="add",
    #                                    mul=mul)
    #         max = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="max", mul=mul)
    #         min = generalized_rspmm(edge_index, edge_type, edge_weight, relation, input, sum="min", mul=mul)
    #         mean = (sum + boundary) / degree_out
    #         sq_mean = (sq_sum + boundary ** 2) / degree_out
    #         max = torch.max(max, boundary)
    #         min = torch.min(min, boundary) # (node, batch_size * input_dim)
    #         std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
    #         features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
    #         features = features.flatten(-2) # (node, batch_size * input_dim * 4)
    #         scale = degree_out.log()
    #         scale = scale / scale.mean() if scale.mean() > 0 else scale
    #         scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1) # (node, 3)
    #         update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2) # (node, batch_size * input_dim * 4 * 3)
    #     else:
    #         raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
    #
    #     update = update.view(num_node, batch_size, -1).transpose(0, 1)
    #     return update

    def update(self, update, input):
        # node update as a function of old states (input) and this layer output (update)
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class DistinctiveGeneralizedRelationalConv(GeneralizedRelationalConv):
    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True,unmasked_tensor=None):
        super(DistinctiveGeneralizedRelationalConv, self).__init__(input_dim, output_dim, num_relation, query_input_dim, message_func,
                 aggregate_func, layer_norm, activation, dependent)
        self.unmasked_tensor=unmasked_tensor
    def forward(self, input, query, boundary, edge_index, edge_type, size, edge_weight=None,query_relations=None):
        batch_size = len(query)


        if self.dependent:
            # layer-specific relation features as a projection of input "query" (relation) embeddings
            relation = self.relation_linear(query).view(batch_size, self.num_relation, self.input_dim)
        else:
            # layer-specific relation features as a special embedding matrix unique to each layer
            relation = self.relation.weight.expand(batch_size, -1, -1)
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_type), device=input.device)


        if torch.isnan(edge_weight).any().item():
            raise ValueError("edge_weight has nan")
        if torch.isnan(relation).any().item():
            raise ValueError("relation has nan")

        output=[]
        next_query_relations=[]
        num_relations=self.unmasked_tensor.shape[0]
        unmasked_idxs = []

        for i in range(batch_size):
            ur = self.unmasked_tensor[query_relations[i]]
            if torch.isnan(ur).any().item():
                raise ValueError("ur has nan")
            ur=torch.sum(ur,dim=0)>0
            if torch.isnan(ur).any().item():
                raise ValueError("ur has nan 2")

            nqr=(torch.nonzero(ur).view(-1)+int(num_relations/2)) % num_relations
            if torch.isnan(nqr).any().item():
                raise ValueError("nqr has nan")
            uidxs = ur[edge_type]
            if torch.isnan(uidxs).any().item():
                raise ValueError("uidxs has nan")
            unmasked_idxs.append(uidxs)
            next_query_relations.append(nqr)

        output = self.propagate(input=input, relation=relation, boundary=boundary, edge_index=edge_index,
                                edge_type=edge_type, size=size, edge_weight=edge_weight,unmasked_idxs=unmasked_idxs)


        # for i in range(batch_size):
        #     unmasked_relations = self.unmasked_tensor[query_relations[i]]
        #     unmasked_relations=torch.sum(unmasked_relations,dim=0)>0
        #
        #     nqr=(torch.nonzero(unmasked_relations).view(-1)+int(num_relations/2)) % num_relations
        #     unmasked_idxs = unmasked_relations[edge_type]
        #     o = self.propagate(input=input[i,None], relation=relation[i,None], boundary=boundary[i,None], edge_index=edge_index[:,unmasked_idxs],
        #                             edge_type=edge_type[unmasked_idxs], size=size, edge_weight=edge_weight[unmasked_idxs])
        #     output.append(o)
        #     next_query_relations.append(nqr)
        # output=torch.cat(output,dim=0)
        return output,next_query_relations


    def message(self, input_j, relation, boundary, edge_type,unmasked_idxs):
        relation_j = relation.index_select(self.node_dim, edge_type)
        if torch.isnan(relation_j).any().item():
            raise ValueError("relation_j has nan")
        if self.message_func == "transe":
            message = input_j + relation_j
        elif self.message_func == "distmult":
            message = input_j * relation_j
        elif self.message_func == "rotate":
            x_j_re, x_j_im = input_j.chunk(2, dim=-1)
            r_j_re, r_j_im = relation_j.chunk(2, dim=-1)
            message_re = x_j_re * r_j_re - x_j_im * r_j_im
            message_im = x_j_re * r_j_im + x_j_im * r_j_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        for i in range(len(unmasked_idxs)):
            idx=unmasked_idxs[i]
            message[i][idx==False,:]=self.eps
        message= torch.cat([message, boundary], dim=self.node_dim)  # (num_edges + num_nodes, batch_size, input_dim)
        if torch.isnan(message).any().item():
            raise ValueError("message has nan")
        return message