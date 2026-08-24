import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from vq import ResidualVectorQuant


class GlobalAttn(torch.nn.Module):
    def __init__(self, hidden_channels, heads, num_layers, beta, dropout, qk_shared=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.num_layers = num_layers
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared

        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(num_layers, heads * hidden_channels))
        else:
            self.betas = torch.nn.Parameter(torch.ones(num_layers, heads * hidden_channels) * self.beta)

        self.h_lins = torch.nn.ModuleList()
        if not self.qk_shared:
            self.q_lins = torch.nn.ModuleList()
        self.k_lins = torch.nn.ModuleList()
        self.v_lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.h_lins.append(torch.nn.Linear(heads * hidden_channels, heads * hidden_channels))
            if not self.qk_shared:
                self.q_lins.append(torch.nn.Linear(heads * hidden_channels, heads * hidden_channels))
            self.k_lins.append(torch.nn.Linear(heads * hidden_channels, heads * hidden_channels))
            self.v_lins.append(torch.nn.Linear(heads * hidden_channels, heads * hidden_channels))
            self.lns.append(torch.nn.LayerNorm(heads * hidden_channels))

        self.lin_out = torch.nn.Linear(heads * hidden_channels, heads * hidden_channels)

    def reset_parameters(self):
        for h_lin in self.h_lins:
            h_lin.reset_parameters()
        if not self.qk_shared:
            for q_lin in self.q_lins:
                q_lin.reset_parameters()
        for k_lin in self.k_lins:
            k_lin.reset_parameters()
        for v_lin in self.v_lins:
            v_lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, x):
        seq_len, _ = x.size()
        for i in range(self.num_layers):
            h = self.h_lins[i](x)
            k = torch.sigmoid(self.k_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            q = k if self.qk_shared else torch.sigmoid(self.q_lins[i](x)).view(seq_len, self.hidden_channels, self.heads)
            v = self.v_lins[i](x).view(seq_len, self.hidden_channels, self.heads)

            kv = torch.einsum('ndh, nmh -> dmh', k, v)
            num = torch.einsum('ndh, dmh -> nmh', q, kv)
            k_sum = torch.einsum('ndh -> dh', k)
            den = torch.einsum('ndh, dh -> nh', q, k_sum).unsqueeze(1)

            beta = torch.sigmoid(self.betas[i]).unsqueeze(0) if self.beta < 0 else self.betas[i].unsqueeze(0)
            x = (num / den).reshape(seq_len, -1)
            x = self.lns[i](x) * (h + beta)
            x = F.relu(self.lin_out(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def build_mlp(in_dim, hidden_dim, out_dim, num_layers, dropout):
    if num_layers <= 1:
        return torch.nn.Linear(in_dim, out_dim)
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
    for _ in range(num_layers - 2):
        layers += [torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
    layers += [torch.nn.Linear(hidden_dim, out_dim)]
    return torch.nn.Sequential(*layers)


class CoDiFormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 local_layers=3, global_layers=2, in_dropout=0.15, dropout=0.5,
                 global_dropout=0.5, heads=1, beta=-1, pre_ln=False,
                 use_nodeid_aux=True, nodeid_pred_mode='gate', use_multi_layer_ids=False,
                 vq_dim=64, num_codes=16, rvq_layers=3, id_emb_dim=16,
                 id_mlp_layers=2, id_mlp_hidden=256, id_mlp_dropout=0.5,
                 id_fused_dim=128, gate_hidden=256):
        super().__init__()
        self._global = False
        self.in_drop = in_dropout
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.use_nodeid_aux = use_nodeid_aux
        self.nodeid_pred_mode = nodeid_pred_mode
        self.use_multi_layer_ids = use_multi_layer_ids
        self.gate_hidden = gate_hidden
        self.local_layers = local_layers
        self.beta = beta

        d_model = heads * hidden_channels
        if self.beta < 0:
            self.betas = torch.nn.Parameter(torch.zeros(local_layers, d_model))
        else:
            self.betas = torch.nn.Parameter(torch.ones(local_layers, d_model) * self.beta)

        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()

        for _ in range(local_layers):
            self.h_lins.append(torch.nn.Linear(d_model, d_model))
            self.local_convs.append(GATConv(d_model, hidden_channels, heads=heads, concat=True, add_self_loops=False, bias=False))
            self.lins.append(torch.nn.Linear(d_model, d_model))
            self.lns.append(torch.nn.LayerNorm(d_model))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(d_model))

        self.lin_in = torch.nn.Linear(in_channels, d_model)
        self.ln = torch.nn.LayerNorm(d_model)
        self.global_attn = GlobalAttn(hidden_channels, heads, global_layers, beta, global_dropout)
        self.pred_local = torch.nn.Linear(d_model, out_channels)
        self.pred_global = torch.nn.Linear(d_model, out_channels)

        self.rvq_layers = rvq_layers
        self.id_emb_dim = id_emb_dim
        self.id_raw_dim = rvq_layers * id_emb_dim

        if self.use_nodeid_aux:
            if self.use_multi_layer_ids:
                self.local_vq_projs = torch.nn.ModuleList([torch.nn.Linear(d_model, vq_dim) for _ in range(local_layers)])
                self.local_rvqs = torch.nn.ModuleList([
                    ResidualVectorQuant(dim=vq_dim, codebook_size=num_codes, num_res_layers=rvq_layers,
                                        decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False)
                    for _ in range(local_layers)
                ])
                self.local_id_embs = torch.nn.ModuleList([
                    torch.nn.ModuleList([torch.nn.Embedding(num_codes, id_emb_dim) for _ in range(rvq_layers)])
                    for _ in range(local_layers)
                ])
                self.global_vq_proj = torch.nn.Linear(d_model, vq_dim)
                self.global_rvq = ResidualVectorQuant(dim=vq_dim, codebook_size=num_codes, num_res_layers=rvq_layers,
                                                      decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False)
                self.global_id_embs = torch.nn.ModuleList([torch.nn.Embedding(num_codes, id_emb_dim) for _ in range(rvq_layers)])
                self.id_layer_projs = torch.nn.ModuleList([torch.nn.Linear(self.id_raw_dim, id_fused_dim) for _ in range(local_layers + 1)])
                self.id_layer_gate = torch.nn.Linear((local_layers + 1) * d_model, local_layers + 1)
                id_feat_dim = id_fused_dim
            else:
                self.vq_proj = torch.nn.Linear(d_model, vq_dim)
                self.rvq = ResidualVectorQuant(dim=vq_dim, codebook_size=num_codes, num_res_layers=rvq_layers,
                                               decay=0.8, commitment_weight=0.25, use_cosine_sim=True, kmeans_init=False)
                self.id_embs = torch.nn.ModuleList([torch.nn.Embedding(num_codes, id_emb_dim) for _ in range(rvq_layers)])
                id_feat_dim = self.id_raw_dim

            self.pred_local_id = torch.nn.Linear(d_model + id_feat_dim, out_channels)
            self.pred_global_id = torch.nn.Linear(d_model + id_feat_dim, out_channels)
            self.pred_local_id_only = build_mlp(id_feat_dim, id_mlp_hidden, out_dim=out_channels, num_layers=id_mlp_layers, dropout=id_mlp_dropout)
            self.pred_global_id_only = build_mlp(id_feat_dim, id_mlp_hidden, out_dim=out_channels, num_layers=id_mlp_layers, dropout=id_mlp_dropout)

            # Heterophily-aware adaptive continuous-discrete fusion gate.
            # It first maps the discrete semantic ID feature to the same dimension as
            # the continuous Transformer representation, then learns a node-wise
            # vector gate to dynamically balance both sources.
            self.id_to_cont = torch.nn.Linear(id_feat_dim, d_model)

            self.cont_id_gate = torch.nn.Sequential(
                torch.nn.Linear(2 * d_model, gate_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(id_mlp_dropout),
                torch.nn.Linear(gate_hidden, d_model),
                torch.nn.Sigmoid(),
            )

            self.pred_local_gate = torch.nn.Linear(d_model, out_channels)
            self.pred_global_gate = torch.nn.Linear(d_model, out_channels)

    def reset_parameters(self):
        for m in self.local_convs:
            m.reset_parameters()
        for m in self.lins:
            m.reset_parameters()
        for m in self.h_lins:
            m.reset_parameters()
        for m in self.lns:
            m.reset_parameters()
        if self.pre_ln:
            for m in self.pre_lns:
                m.reset_parameters()
        self.lin_in.reset_parameters()
        self.ln.reset_parameters()
        self.global_attn.reset_parameters()
        self.pred_local.reset_parameters()
        self.pred_global.reset_parameters()

        if self.use_nodeid_aux:
            if self.use_multi_layer_ids:
                for p in self.local_vq_projs:
                    p.reset_parameters()
                self.global_vq_proj.reset_parameters()
                for layer_embs in self.local_id_embs:
                    for emb in layer_embs:
                        torch.nn.init.xavier_uniform_(emb.weight)
                for emb in self.global_id_embs:
                    torch.nn.init.xavier_uniform_(emb.weight)
                for p in self.id_layer_projs:
                    p.reset_parameters()
                self.id_layer_gate.reset_parameters()
            else:
                self.vq_proj.reset_parameters()
                for emb in self.id_embs:
                    torch.nn.init.xavier_uniform_(emb.weight)
            self.pred_local_id.reset_parameters()
            self.pred_global_id.reset_parameters()
            self.id_to_cont.reset_parameters()

            for m in self.cont_id_gate.modules():
                if isinstance(m, torch.nn.Linear):
                    m.reset_parameters()

            self.pred_local_gate.reset_parameters()
            self.pred_global_gate.reset_parameters()
            for m in self.pred_local_id_only.modules():
                if isinstance(m, torch.nn.Linear):
                    m.reset_parameters()
            for m in self.pred_global_id_only.modules():
                if isinstance(m, torch.nn.Linear):
                    m.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    @staticmethod
    def _ids_to_feat(id_list, emb_tables):
        return torch.cat([emb(ids) for ids, emb in zip(id_list, emb_tables)], dim=-1)

    def _fuse_multi_layer_ids(self, local_hiddens, attn_out, id_feats):
        cont_ctx = torch.cat(local_hiddens + [attn_out], dim=-1)
        alpha = torch.softmax(self.id_layer_gate(cont_ctx), dim=-1)
        proj_feats = [proj(feat) for proj, feat in zip(self.id_layer_projs, id_feats)]
        fused = 0
        for i, pf in enumerate(proj_feats):
            fused = fused + alpha[:, i:i + 1] * pf
        return fused

    def _adaptive_fuse_cont_id(self, cont_feat, id_feat):
        id_proj = self.id_to_cont(id_feat)
        gate = self.cont_id_gate(torch.cat([cont_feat, id_proj], dim=-1))
        fused = gate * cont_feat + (1.0 - gate) * id_proj
        return fused, gate

    def forward(self, x, edge_index, return_aux=False):
        x = F.dropout(x, p=self.in_drop, training=self.training)
        x = self.lin_in(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_local = 0
        local_hiddens = []
        for i, conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            h = F.relu(self.h_lins[i](x))
            x = conv(x, edge_index) + self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            beta = torch.sigmoid(self.betas[i]).unsqueeze(0) if self.beta < 0 else self.betas[i].unsqueeze(0)
            x = (1 - beta) * self.lns[i](h * x) + beta * x
            x_local = x_local + x
            local_hiddens.append(x)

        attn_out = self.global_attn(self.ln(x_local)) if self._global else x_local

        if self.use_nodeid_aux:
            if self.use_multi_layer_ids:
                id_feats, id_tensors, vq_losses = [], [], []
                for h_local, proj, rvq, emb_tables in zip(local_hiddens, self.local_vq_projs, self.local_rvqs, self.local_id_embs):
                    z = proj(h_local)
                    _, id_list, vq_loss, _, _ = rvq(z)
                    id_feats.append(self._ids_to_feat(id_list, emb_tables))
                    id_tensors.append(torch.stack(id_list, dim=1))
                    vq_losses.append(vq_loss.mean())
                z_g = self.global_vq_proj(attn_out)
                _, id_list_g, vq_loss_g, _, _ = self.global_rvq(z_g)
                id_feats.append(self._ids_to_feat(id_list_g, self.global_id_embs))
                id_tensors.append(torch.stack(id_list_g, dim=1))
                vq_losses.append(vq_loss_g.mean())
                node_id_feat = self._fuse_multi_layer_ids(local_hiddens, attn_out, id_feats)
                node_ids = torch.cat(id_tensors, dim=1).to(torch.int32)
                vq_loss = torch.stack(vq_losses).mean()
            else:
                z = self.vq_proj(attn_out)
                _, id_list, vq_loss, _, _ = self.rvq(z)
                node_id_feat = self._ids_to_feat(id_list, self.id_embs)
                node_ids = torch.stack(id_list, dim=1).to(torch.int32)
                vq_loss = vq_loss.mean()

            cont_logits = self.pred_global(attn_out) if self._global else self.pred_local(attn_out)
            id_logits = self.pred_global_id_only(node_id_feat) if self._global else self.pred_local_id_only(node_id_feat)
            gate = None
            if self.nodeid_pred_mode == "cont_only":
                out = cont_logits

            elif self.nodeid_pred_mode == "id_only":
                out = id_logits

            elif self.nodeid_pred_mode == "concat":
                pred_in = torch.cat([attn_out, node_id_feat], dim=-1)
                out = (
                    self.pred_global_id(pred_in)
                    if self._global
                    else self.pred_local_id(pred_in)
                )

            elif self.nodeid_pred_mode == "gate":
                fused_feat, gate = self._adaptive_fuse_cont_id(
                    attn_out,
                    node_id_feat
                )
                out = (
                    self.pred_global_gate(fused_feat)
                    if self._global
                    else self.pred_local_gate(fused_feat)
                )

            else:
                raise ValueError(
                    f"Unknown nodeid_pred_mode: {self.nodeid_pred_mode}"
                )
        else:
            cont_logits = self.pred_global(attn_out) if self._global else self.pred_local(attn_out)
            id_logits = None
            out = cont_logits
            vq_loss = x.new_zeros(())
            node_ids = None
            gate = None

        if return_aux:
            return {'logits': out, 'cont_logits': cont_logits, 'id_logits': id_logits, 'vq_loss': vq_loss, 'node_ids': node_ids, 'fusion_gate': gate}
        return out
