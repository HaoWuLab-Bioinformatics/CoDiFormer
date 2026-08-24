import torch
import torch.nn.functional as F
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS, Planetoid
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def load_fixed_splits(data_dir, dataset, name):
    splits_lst = []
    if name in ['cora', 'citeseer', 'pubmed']:
        pyg_name = {
            'cora': 'Cora',
            'citeseer': 'CiteSeer',
            'pubmed': 'PubMed',
        }[name]
        torch_dataset = Planetoid(root=f'{data_dir}/Planetoid', name=pyg_name, split='public')
        data = torch_dataset[0]
        splits = {}
        splits['train'] = torch.where(data.train_mask)[0]
        splits['valid'] = torch.where(data.val_mask)[0]
        splits['test'] = torch.where(data.test_mask)[0]
        splits_lst.append(splits)
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:, i])[0]
            splits['valid'] = torch.where(data.val_mask[:, i])[0]
            splits['test'] = torch.where(data.test_mask[:, i])[0]
            splits_lst.append(splits)
    elif name in ['wikics']:
        torch_dataset = WikiCS(root=f"{data_dir}/wikics/")
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:, i])[0]
            splits['valid'] = torch.where(torch.logical_or(data.val_mask, data.stopping_mask)[:, i])[0]
            splits['test'] = torch.where(data.test_mask[:])[0]
            splits_lst.append(splits)
    elif name in ['amazon-computer', 'amazon-photo', 'coauthor-cs', 'coauthor-physics']:
        splits = {}
        idx = np.load(f'{data_dir}/{name}_split.npz')
        splits['train'] = torch.from_numpy(idx['train'])
        splits['valid'] = torch.from_numpy(idx['valid'])
        splits['test'] = torch.from_numpy(idx['test'])
        splits_lst.append(splits)
    elif name in ['pokec']:
        split = np.load(f'{data_dir}/{name}/{name}-splits.npy', allow_pickle=True)
        for i in range(split.shape[0]):
            splits = {}
            splits['train'] = torch.from_numpy(np.asarray(split[i]['train']))
            splits['valid'] = torch.from_numpy(np.asarray(split[i]['valid']))
            splits['test'] = torch.from_numpy(np.asarray(split[i]['test']))
            splits_lst.append(splits)
    elif name in ['ogbn-arxiv', 'ogbn-products'] and hasattr(dataset, 'load_fixed_splits'):
        splits = dataset.load_fixed_splits()
        splits_lst.append(splits)
    elif name in ["chameleon", "squirrel"]:
        file_path = f"{data_dir}/geom-gcn/{name}/{name}_filtered.npz"
        data = np.load(file_path)
        train_masks = data["train_masks"]  # (10, N), 10 splits
        val_masks = data["val_masks"]
        test_masks = data["test_masks"]
        N = train_masks.shape[1]

        node_idx = np.arange(N)
        for i in range(10):
            splits = {}
            splits["train"] = torch.as_tensor(node_idx[train_masks[i]])
            splits["valid"] = torch.as_tensor(node_idx[val_masks[i]])
            splits["test"] = torch.as_tensor(node_idx[test_masks[i]])
            splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst

def _to_numpy_labels(y_true):
    y_true = y_true.detach().cpu().numpy()
    if y_true.ndim == 0:
        y_true = y_true.reshape(1)
    return y_true


def eval_f1(y_true, y_pred):
    y_true = _to_numpy_labels(y_true)
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy().reshape(-1)

    if y_true.ndim == 1:
        return f1_score(y_true, y_pred, average='macro')

    acc_list = []
    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true[:, i], y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)


def eval_acc(y_true, y_pred):
    y_true = _to_numpy_labels(y_true)
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    if y_true.ndim == 1:
        return float((y_true == y_pred.reshape(-1)).mean())

    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    rocauc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)


dataset_drive_url = {
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}
