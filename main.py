import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from datetime import datetime
import os

from stability import NodeNorm, PairNorm, LabelSmoothingCrossEntropy
from logger import save_result
from dataset import load_dataset
from eval import eval_acc, eval_rocauc, evaluate
from data_utils import load_fixed_splits
from parse import parse_method, parser_add_main_args


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def distill_kl_loss(teacher_logits, student_logits, temperature=2.0):
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (temperature ** 2)


parser = argparse.ArgumentParser(description='CoDiFormer + RVQ + continuous->ID distillation (_c)')
parser_add_main_args(parser)
args = parser.parse_args()

if not args.global_dropout:
    args.global_dropout = args.dropout

print(args)
fix_seed(args.seed)

device = torch.device('cpu') if args.cpu else (torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device('cpu'))

dataset = load_dataset(args.data_dir, args.dataset)
if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)
split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]
print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
dataset.graph['edge_index'], dataset.graph['node_feat'] = dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

model = parse_method(args, n, c, d, device)

if args.dataset in ('questions'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing) if args.label_smoothing > 0 else nn.NLLLoss()

eval_func = eval_rocauc if args.metric == 'rocauc' else eval_acc


in_norm = None
if args.input_norm == 'nodenorm':
    in_norm = NodeNorm().to(device)
elif args.input_norm == 'pairnorm':
    in_norm = PairNorm().to(device)

log_dir = os.path.join('logs')
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_f = open(log_path, 'a', encoding='utf-8')

def log_print(msg):
    print(msg)
    log_f.write(msg + '\n')
    log_f.flush()


model.train()
log_print('MODEL: ' + str(model))

for run in range(args.runs):
    split_idx = split_idx_lst[0] if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo') else split_idx_lst[run % len(split_idx_lst)]
    train_idx = split_idx['train'].to(device)

    log_print(f'Run {run + 1:02d} start')

    model.reset_parameters()
    model._global = False
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    best_val, best_test = float('-inf'), float('-inf')

    for epoch in range(args.local_epochs + args.global_epochs):
        if epoch == args.local_epochs:
            print('start global attention!!!!!!')
            model._global = True

        model.train()
        optimizer.zero_grad()

        x_in = dataset.graph['node_feat']
        if in_norm is not None:
            x_in = in_norm(x_in)

        out_dict = model(x_in, dataset.graph['edge_index'], return_aux=True)
        logits, cont_logits, id_logits = out_dict['logits'], out_dict['cont_logits'], out_dict['id_logits']
        vq_loss = out_dict['vq_loss']

        if args.dataset in ('questions'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            y = true_label.squeeze(1)[train_idx].to(torch.float)
            loss_main = criterion(logits[train_idx], y)
            loss_cont = criterion(cont_logits[train_idx], y)
        else:
            y = dataset.label.squeeze(1)[train_idx]
            if isinstance(criterion, LabelSmoothingCrossEntropy):
                loss_main = criterion(logits[train_idx], y)
                loss_cont = criterion(cont_logits[train_idx], y)
            else:
                loss_main = criterion(F.log_softmax(logits, dim=1)[train_idx], y)
                loss_cont = criterion(F.log_softmax(cont_logits, dim=1)[train_idx], y)

        if id_logits is not None:
            if args.dataset in ('questions'):
                loss_id = criterion(id_logits[train_idx], y)
            else:
                if isinstance(criterion, LabelSmoothingCrossEntropy):
                    loss_id = criterion(id_logits[train_idx], y)
                else:
                    loss_id = criterion(F.log_softmax(id_logits, dim=1)[train_idx], y)
            loss_distill = distill_kl_loss(cont_logits[train_idx].detach(), id_logits[train_idx], temperature=args.distill_temp)
        else:
            loss_id = logits.new_zeros(())
            loss_distill = logits.new_zeros(())

        if args.nodeid_pred_mode == 'id_only':
            if args.train_with_cont_aux:
                loss_cls = loss_main + loss_cont + args.distill_weight * loss_distill
            else:
                loss_cls = loss_main + args.distill_weight * loss_distill
        else:
            if args.train_with_cont_aux:
                loss_cls = loss_main + loss_cont + args.id_aux_weight * loss_id + args.distill_weight * loss_distill
            else:
                loss_cls = loss_main + args.id_aux_weight * loss_id + args.distill_weight * loss_distill

        loss = loss_cls + args.vq_weight * vq_loss
        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)
        if len(result) == 5:
            train_acc, valid_acc, test_acc, valid_loss, out = result
        else:
            train_acc, valid_acc, test_acc, valid_loss, out, extra = result


        if valid_acc > best_val:
            best_val, best_test = valid_acc, test_acc

        if epoch % args.display_step == 0:
            log_print(
                f"Epoch: {epoch:02d}, Loss: {loss.item():.4f}, "
                f"Train: {100 * train_acc:.2f}%, "
                f"Valid: {100 * valid_acc:.2f}%, "
                f"Test: {100 * test_acc:.2f}%, "
                f"Best Valid: {100 * best_val:.2f}%, "
                f"Best Test: {100 * best_test:.2f}%"
            )
    save_result(args, f'Run {run + 1} finished, Best Test: {best_test:.4f}')


log_f.close()