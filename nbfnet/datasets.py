import os

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url

from collections import defaultdict
import networkx as nx

class IndRelLinkPredDataset(InMemoryDataset):

    urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
        ],
        "NELL-995": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/valid.txt"
        ]
    }

    def __init__(self, root, name, version, transform=None, pre_transform=None):
        self.name = name
        self.version = version
        assert name in ["FB15k-237", "WN18RR", "NELL-995"]
        assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root,'inductive', self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root,'inductive', self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name

class TransRelLinkPredDataset(InMemoryDataset):

    urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt",

        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt",

        ],
        "NELL-995": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/nell_%s/valid.txt",
        ]
    }

    def __init__(self, root, name, version, transform=None, pre_transform=None):
        self.name = name
        self.version = version
        assert name in ["FB15k-237", "WN18RR", "NELL-995"]
        assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        return int(self.data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root,'transductive', self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root,'transductive', self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
             "test.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:1]
        train_files = self.raw_paths[1:]

        inv_train_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    h = inv_train_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(None, sum(num_samples[:1]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:2]), sum(num_samples))
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_train_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name


class DistinctiveIndRelLinkPredDataset(IndRelLinkPredDataset):
    def __init__(self, root, name, version,device=None, transform=None, pre_transform=None,accuracy_threshold=None,recall_threshold=None):
        self.accuracy_threshold=accuracy_threshold
        self.recall_threshold=recall_threshold
        self.name = name
        self.version = version
        self.device=device
        assert name in ["FB15k-237", "WN18RR", "NELL-995"]
        assert version in ["v1", "v2", "v3", "v4"]
        super(IndRelLinkPredDataset,self).__init__(root, transform, pre_transform)
        self.process()
        self.data_and_slices,self.accuracy_tensor,self.recall_tensor = torch.load(self.processed_paths[0],map_location=self.device)
        self.data,self.slices=self.data_and_slices

    @property
    def processed_file_names(self):
        return "data_distinctive.pt"
    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []


        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        self.relation2id=inv_relation_vocab.copy()
        self.entity2id=inv_train_entity_vocab.copy()
        num_relations=len(self.relation2id)
        for r in inv_relation_vocab:
            self.relation2id[r + '_inv'] = inv_relation_vocab[r] + num_relations
        accuracy_tensor, recall_tensor=self.generate_distinctive_neighbor(self.raw_dir,self.processed_dir,self.accuracy_threshold,self.recall_threshold)


        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data]),accuracy_tensor,recall_tensor), self.processed_paths[0])


    def generate_distinctive_neighbor(self, dir, save_dir, accuracy_threshold, recall_threshold):


        triplets = []
        relations = set()
        G_train = nx.DiGraph()
        with open(os.path.join(dir, 'train.txt')) as f:
            for line in f:
                h, r, t = line.strip().split()
                triplets.append([h, r, t])
                triplets.append([t, r + '_inv', h])
                G_train.add_edge(h, t, relation=r)
                G_train.add_edge(t, h, relation=str(r + '_inv'))
                relations.add(r)
                relations.add(r + '_inv')


        relation2neighbors = defaultdict(lambda: defaultdict(int))
        all_neighbors = defaultdict(int)
        for u in G_train.nodes():
            neighbors = frozenset(G_train[u][v]['relation'] for v in G_train[u])
            all_neighbors[neighbors] += 1
            for r in sorted(neighbors):
                relation2neighbors[r][neighbors] += 1

        neighbor_num = defaultdict(int)
        for r in sorted(relations):
            neighbor_num[r] = sum([relation2neighbors[r][n] for n in relation2neighbors[r]])

        accuracy_neighbors = defaultdict(set)
        recall_neighbors = defaultdict(set)
        for r in sorted(relations):
            for r2 in sorted(relations):
                cooccurrence = 0
                for n1 in relation2neighbors[r]:
                    if r2 in n1:
                        cooccurrence += relation2neighbors[r][n1]
                # cooccurrence2 = 0
                # for n2 in relation2neighbors[r2]:
                #     if r in n2:
                #         cooccurrence2+=relation2neighbors[r][n2]
                # assert cooccurrence2==cooccurrence
                if neighbor_num[r] > 0 and cooccurrence / neighbor_num[r] > accuracy_threshold:
                    accuracy_neighbors[r].add(r2)
                if neighbor_num[r2] > 0 and cooccurrence / neighbor_num[r2] > recall_threshold:
                    recall_neighbors[r].add(r2)

        accuracy_neighbors = {r: sorted(accuracy_neighbors[r]) for r in
                              sorted(relations)}
        recall_neighbors = {r: sorted(recall_neighbors[r]) for r in
                            sorted(relations)}

        # accuracy_neighbors = {self.relation2id[r]: torch.tensor(
        #     [self.relation2id[accuracy_neighbors[r][i]] for i in range(len(accuracy_neighbors[r]))]) for r in
        #     accuracy_neighbors}
        # recall_neighbors = {self.relation2id[r]: torch.tensor(
        #     [self.relation2id[recall_neighbors[r][i]] for i in range(len(recall_neighbors[r]))]) for
        #     r in recall_neighbors}

        accuracy_tensor_indices = torch.tensor([[self.relation2id[r2], self.relation2id[r1]] for r2 in
                                                sorted(relations) for r1 in accuracy_neighbors[r2]]).to(self.device)
        recall_tensor_indices = torch.tensor([[self.relation2id[r2], self.relation2id[r1]] for r2 in
                                              sorted(relations) for r1 in recall_neighbors[r2]]).to(self.device)
        accuracy_tensor = torch.zeros([len(relations), len(relations)], dtype=torch.bool).to(self.device)
        recall_tensor = torch.zeros([len(relations), len(relations)], dtype=torch.bool).to(self.device)
        if accuracy_tensor_indices.shape[0] > 0:
            accuracy_tensor[accuracy_tensor_indices[:, 0], accuracy_tensor_indices[:, 1]] = True
        if recall_tensor_indices.shape[0] > 0:
            recall_tensor[recall_tensor_indices[:, 0], recall_tensor_indices[:, 1]] = True

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "distinctive_neighbors.pkl"), "wb") as f:
            torch.save([accuracy_tensor, recall_tensor], f)
        return accuracy_tensor, recall_tensor


class DistinctiveTransRelLinkPredDataset(TransRelLinkPredDataset):
    def __init__(self, root, name, version,device=None, transform=None, pre_transform=None,accuracy_threshold=None,recall_threshold=None):
        self.accuracy_threshold=accuracy_threshold
        self.recall_threshold=recall_threshold
        self.name = name
        self.version = version
        self.device=device
        assert name in ["FB15k-237", "WN18RR", "NELL-995"]
        assert version in ["v1", "v2", "v3", "v4"]
        super(TransRelLinkPredDataset,self).__init__(root, transform, pre_transform)
        self.process()
        self.data_and_slices,self.accuracy_tensor,self.recall_tensor = torch.load(self.processed_paths[0],map_location=self.device)
        self.data,self.slices=self.data_and_slices

    @property
    def processed_file_names(self):
        return "data_distinctive.pt"
    def process(self):
        test_files = self.raw_paths[:1]
        train_files = self.raw_paths[1:]

        inv_train_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []
        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    h = inv_train_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        self.relation2id=inv_relation_vocab.copy()
        self.entity2id=inv_train_entity_vocab.copy()
        num_relations=len(self.relation2id)
        for r in inv_relation_vocab:
            self.relation2id[r + '_inv'] = inv_relation_vocab[r] + num_relations
        accuracy_tensor, recall_tensor=self.generate_distinctive_neighbor(self.raw_dir,self.processed_dir,self.accuracy_threshold,self.recall_threshold)


        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(None, sum(num_samples[:1]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:2]), sum(num_samples))
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_train_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data]),accuracy_tensor,recall_tensor), self.processed_paths[0])


    def generate_distinctive_neighbor(self, dir, save_dir, accuracy_threshold, recall_threshold):


        triplets = []
        relations = set()
        G_train = nx.DiGraph()
        with open(os.path.join(dir, 'train.txt')) as f:
            for line in f:
                h, r, t = line.strip().split()
                triplets.append([h, r, t])
                triplets.append([t, r + '_inv', h])
                G_train.add_edge(h, t, relation=r)
                G_train.add_edge(t, h, relation=str(r + '_inv'))
                relations.add(r)
                relations.add(r + '_inv')




        relation2neighbors = defaultdict(lambda: defaultdict(int))
        all_neighbors = defaultdict(int)
        for u in G_train.nodes():
            neighbors = frozenset(G_train[u][v]['relation'] for v in G_train[u])
            all_neighbors[neighbors] += 1
            for r in sorted(neighbors):
                relation2neighbors[r][neighbors] += 1

        neighbor_num = defaultdict(int)
        for r in sorted(relations):
            neighbor_num[r] = sum([relation2neighbors[r][n] for n in relation2neighbors[r]])

        accuracy_neighbors = defaultdict(set)
        recall_neighbors = defaultdict(set)
        for r in sorted(relations):
            for r2 in sorted(relations):
                cooccurrence = 0
                for n1 in relation2neighbors[r]:
                    if r2 in n1:
                        cooccurrence += relation2neighbors[r][n1]
                # cooccurrence2 = 0
                # for n2 in relation2neighbors[r2]:
                #     if r in n2:
                #         cooccurrence2+=relation2neighbors[r][n2]
                # assert cooccurrence2==cooccurrence
                if neighbor_num[r] > 0 and cooccurrence / neighbor_num[r] > accuracy_threshold:
                    accuracy_neighbors[r].add(r2)
                if neighbor_num[r2] > 0 and cooccurrence / neighbor_num[r2] > recall_threshold:
                    recall_neighbors[r].add(r2)

        accuracy_neighbors = {r: sorted(accuracy_neighbors[r]) for r in
                              sorted(relations)}
        recall_neighbors = {r: sorted(recall_neighbors[r]) for r in
                            sorted(relations)}

        # accuracy_neighbors = {self.relation2id[r]: torch.tensor(
        #     [self.relation2id[accuracy_neighbors[r][i]] for i in range(len(accuracy_neighbors[r]))]) for r in
        #     accuracy_neighbors}
        # recall_neighbors = {self.relation2id[r]: torch.tensor(
        #     [self.relation2id[recall_neighbors[r][i]] for i in range(len(recall_neighbors[r]))]) for
        #     r in recall_neighbors}

        accuracy_tensor_indices = torch.tensor([[self.relation2id[r2], self.relation2id[r1]] for r2 in
                                                sorted(relations) for r1 in accuracy_neighbors[r2]]).to(self.device)
        recall_tensor_indices = torch.tensor([[self.relation2id[r2], self.relation2id[r1]] for r2 in
                                              sorted(relations) for r1 in recall_neighbors[r2]]).to(self.device)
        accuracy_tensor = torch.zeros([len(relations), len(relations)], dtype=torch.bool).to(self.device)
        recall_tensor = torch.zeros([len(relations), len(relations)], dtype=torch.bool).to(self.device)
        if accuracy_tensor_indices.shape[0] > 0:
            accuracy_tensor[accuracy_tensor_indices[:, 0], accuracy_tensor_indices[:, 1]] = True
        if recall_tensor_indices.shape[0] > 0:
            recall_tensor[recall_tensor_indices[:, 0], recall_tensor_indices[:, 1]] = True

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, "distinctive_neighbors.pkl"), "wb") as f:
            torch.save([accuracy_tensor, recall_tensor], f)
        return accuracy_tensor, recall_tensor
