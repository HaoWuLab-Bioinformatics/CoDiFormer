import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from stability import NodeNorm, PairNorm, LabelSmoothingCrossEntropy


def _prepare_targets(y):
    if y.dim() == 0:
        return y.reshape(1)
    if y.dim() > 1 and y.shape[-1] == 1:
        return y.reshape(-1)
    return y


def _to_numpy_label(y):
    return _prepare_targets(y).detach().cpu().numpy()


def _to_numpy_pred_label(y_pred):
    return y_pred.detach().cpu().argmax(dim=-1).numpy().reshape(-1)


def eval_acc(y_true, y_pred):
    y_true = _prepare_targets(y_true).detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    if y_true.ndim == 1:
        return float((y_true == y_pred.reshape(-1)).mean())

    acc_list = []
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(correct.mean()) if len(correct) > 0 else 0.0)

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    """兼容原来的 eval_func 调用。无法计算 ROC-AUC 时返回 nan，而不是中断训练。"""
    return _safe_roc_auc(y_true, y_pred)


def _classification_metrics(y_true, y_pred):
    """
    返回单个 split 上的 Precision / Recall / ROC-AUC / Macro-F1 / Weighted-F1。
    - 普通多分类节点分类：用 argmax 作为预测类别。
    - questions / 多标签形式：若 y_true 是二维，则用 sigmoid > 0.5 得到预测标签。
    - ROC-AUC 在某个 split 只有单一类别时无法计算，此时返回 nan，避免训练中断。
    """
    y_true_np = _to_numpy_label(y_true)
    logits = y_pred.detach().cpu()

    if y_true_np.ndim == 1:
        y_true_flat = y_true_np.reshape(-1)
        y_pred_flat = logits.argmax(dim=-1).numpy().reshape(-1)

        precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
        recall = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        roc_auc = _safe_roc_auc(torch.as_tensor(y_true_flat), y_pred)

    else:
        # 多标签 / one-hot 形式。对每个标签做平均。
        prob = torch.sigmoid(logits).numpy()
        y_pred_bin = (prob >= 0.5).astype(int)

        precision = precision_score(y_true_np, y_pred_bin, average='macro', zero_division=0)
        recall = recall_score(y_true_np, y_pred_bin, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true_np, y_pred_bin, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true_np, y_pred_bin, average='weighted', zero_division=0)
        roc_auc = _safe_roc_auc(torch.as_tensor(y_true_np), y_pred)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc) if roc_auc == roc_auc else float('nan'),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
    }


def _safe_roc_auc(y_true, y_pred):
    """
    安全计算 ROC-AUC：
    1. 二分类标签 + 两列 logits：取 softmax 后正类概率。
    2. 多分类标签 + 多列 logits：使用 macro OVR ROC-AUC。
    3. 多标签标签矩阵：逐标签计算平均 ROC-AUC。
    无法计算时返回 nan。
    """
    y_true_np = _to_numpy_label(y_true)
    logits = y_pred.detach().cpu()

    try:
        if y_true_np.ndim == 1:
            y_true_flat = y_true_np.reshape(-1)
            unique_classes = np.unique(y_true_flat[~np.isnan(y_true_flat)])

            if len(unique_classes) < 2:
                return float('nan')

            if logits.dim() == 1 or logits.size(-1) == 1:
                scores = torch.sigmoid(logits.reshape(-1)).numpy()
                return float(roc_auc_score(y_true_flat, scores))

            prob = F.softmax(logits, dim=-1).numpy()

            if prob.shape[1] == 2:
                return float(roc_auc_score(y_true_flat, prob[:, 1]))

            # 多分类 split 内如果类别不全，sklearn 可能报错；这里捕获后返回 nan。
            return float(roc_auc_score(y_true_flat, prob, multi_class='ovr', average='macro'))

        # 多标签 / one-hot
        if y_true_np.shape[1] == 1:
            return _safe_roc_auc(torch.as_tensor(y_true_np.reshape(-1)), y_pred)

        prob = torch.sigmoid(logits).numpy()
        rocauc_list = []
        for i in range(y_true_np.shape[1]):
            is_labeled = y_true_np[:, i] == y_true_np[:, i]
            yi = y_true_np[is_labeled, i]
            pi = prob[is_labeled, i]
            if np.sum(yi == 1) > 0 and np.sum(yi == 0) > 0:
                rocauc_list.append(roc_auc_score(yi, pi))
        return float(np.mean(rocauc_list)) if rocauc_list else float('nan')
    except Exception:
        return float('nan')


def _merge_split_metrics(train_metrics, valid_metrics, test_metrics):
    extra = {}
    for prefix, metrics in [('train', train_metrics), ('valid', valid_metrics), ('test', test_metrics)]:
        for name, value in metrics.items():
            extra[f'{prefix}_{name}'] = value
    return extra


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        x_in = dataset.graph['node_feat']
        if getattr(args, 'input_norm', 'none') == 'nodenorm':
            x_in = NodeNorm()(x_in)
        elif getattr(args, 'input_norm', 'none') == 'pairnorm':
            x_in = PairNorm()(x_in)

        # 兼容该 VQ/NodeID 模型：优先取 logits，避免返回 dict 时后续指标计算失败。
        out_raw = model(x_in, dataset.graph['edge_index'], return_aux=True)
        out = out_raw['logits'] if isinstance(out_raw, dict) and 'logits' in out_raw else out_raw

    y_metric = _prepare_targets(dataset.label)
    y_train = _prepare_targets(y_metric[split_idx['train']])
    y_valid = _prepare_targets(y_metric[split_idx['valid']])
    y_test = _prepare_targets(y_metric[split_idx['test']])

    train_acc = eval_func(y_train, out[split_idx['train']])
    valid_acc = eval_func(y_valid, out[split_idx['valid']])
    test_acc = eval_func(y_test, out[split_idx['test']])

    train_metrics = _classification_metrics(y_train, out[split_idx['train']])
    valid_metrics = _classification_metrics(y_valid, out[split_idx['valid']])
    test_metrics = _classification_metrics(y_test, out[split_idx['test']])
    extra = _merge_split_metrics(train_metrics, valid_metrics, test_metrics)

    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[split_idx['valid']].to(torch.float))
    else:
        if isinstance(criterion, LabelSmoothingCrossEntropy):
            valid_loss = criterion(out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
        else:
            valid_loss = criterion(F.log_softmax(out, dim=1)[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out, extra


@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None):
    if result is not None:
        out = result
    else:
        model.eval()

        model.to(torch.device('cpu'))
        dataset.label = dataset.label.to(torch.device('cpu'))
        edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
        out_raw = model(x, edge_index, return_aux=True)
        out = out_raw['logits'] if isinstance(out_raw, dict) and 'logits' in out_raw else out_raw

    y_metric = _prepare_targets(dataset.label)
    y_train = _prepare_targets(y_metric[split_idx['train']])
    y_valid = _prepare_targets(y_metric[split_idx['valid']])
    y_test = _prepare_targets(y_metric[split_idx['test']])

    train_acc = eval_func(y_train, out[split_idx['train']])
    valid_acc = eval_func(y_valid, out[split_idx['valid']])
    test_acc = eval_func(y_test, out[split_idx['test']])

    train_metrics = _classification_metrics(y_train, out[split_idx['train']])
    valid_metrics = _classification_metrics(y_valid, out[split_idx['valid']])
    test_metrics = _classification_metrics(y_test, out[split_idx['test']])
    extra = _merge_split_metrics(train_metrics, valid_metrics, test_metrics)

    if args.dataset in ('questions'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[split_idx['valid']].to(torch.float))
    else:
        if isinstance(criterion, LabelSmoothingCrossEntropy):
            valid_loss = criterion(out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
        else:
            valid_loss = criterion(F.log_softmax(out, dim=1)[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out, extra
