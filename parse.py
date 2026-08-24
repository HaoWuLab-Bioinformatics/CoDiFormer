from model import CoDiFormer


def parse_method(args, n, c, d, device):
    model = CoDiFormer(
        d, args.hidden_channels, c,
        local_layers=args.local_layers,
        global_layers=args.global_layers,
        in_dropout=args.in_dropout,
        dropout=args.dropout,
        global_dropout=args.global_dropout,
        heads=args.num_heads,
        beta=args.beta,
        pre_ln=args.pre_ln,
        use_nodeid_aux=args.use_nodeid_aux,
        nodeid_pred_mode=args.nodeid_pred_mode,
        use_multi_layer_ids=args.use_multi_layer_ids,
        vq_dim=args.vq_dim,
        num_codes=args.num_codes,
        rvq_layers=args.rvq_layers,
        id_emb_dim=args.id_emb_dim,
        id_mlp_layers=args.id_mlp_layers,
        id_mlp_hidden=args.id_mlp_hidden,
        id_mlp_dropout=args.id_mlp_dropout,
        id_fused_dim=args.id_fused_dim,
    ).to(device)
    return model


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='roman-empire')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--local_epochs', type=int, default=1000)
    parser.add_argument('--global_epochs', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'])

    parser.add_argument('--method', type=str, default='CoDiFormer')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7)
    parser.add_argument('--global_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--beta', type=float, default=-1.0)
    parser.add_argument('--pre_ln', action='store_true')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--global_dropout', type=float, default=None)

    parser.add_argument('--display_step', type=int, default=100)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--model_dir', type=str, default='./model/')
    parser.add_argument('--save_result', action='store_true')

    parser.add_argument('--exp_name', type=str, default='default',
                        help='experiment name for distinguishing different settings')

    parser.add_argument('--input_norm', type=str, default='none', choices=['none', 'nodenorm', 'pairnorm'],
                        help='normalize input node features before forward')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='label smoothing for non-questions datasets')

    parser.add_argument(
        "--nodeid_pred_mode",
        type=str,
        default="gate",
        choices=[
            "cont_only",
            "concat",
            "gate",
            "id_only",
        ],
        help="Fusion strategy between continuous and semantic ID features."
    )

    parser.add_argument('--use_nodeid_aux', action='store_true',
                        help='enable RVQ node-id branch')
    parser.add_argument('--use_multi_layer_ids', action='store_true',
                        help='use local multi-layer IDs plus global ID')
    parser.add_argument('--vq_dim', type=int, default=64,
                        help='projection dim before RVQ')
    parser.add_argument('--num_codes', type=int, default=16,
                        help='codebook size for each RVQ level')
    parser.add_argument('--rvq_layers', type=int, default=3,
                        help='number of residual quantizers per layer')
    parser.add_argument('--id_emb_dim', type=int, default=16,
                        help='embedding dim for each discrete code')
    parser.add_argument('--vq_weight', type=float, default=0.05,
                        help='weight of RVQ commitment loss')
    parser.add_argument('--nodeid_save_prefix', type=str, default='semantic_ID',
                        help='prefix for saved NodeID npz files')

    parser.add_argument('--id_mlp_layers', type=int, default=2,
                        help='MLP depth for ID-only classifier')
    parser.add_argument('--id_mlp_hidden', type=int, default=256,
                        help='hidden width of ID-only classifier MLP')
    parser.add_argument('--id_mlp_dropout', type=float, default=0.5,
                        help='dropout inside ID-only classifier MLP')
    parser.add_argument('--id_aux_weight', type=float, default=0.3,
                        help='auxiliary loss weight for ID branch when nodeid_pred_mode=concat')
    parser.add_argument('--train_with_cont_aux', action='store_true',
                        help='keep continuous-head auxiliary loss during training (default: True)')
    parser.add_argument('--no_train_with_cont_aux', action='store_false', dest='train_with_cont_aux',
                        help='disable continuous-head auxiliary loss; keep only main + ID/distill/VQ terms')
    parser.set_defaults(train_with_cont_aux=True)
    parser.add_argument('--id_fused_dim', type=int, default=128,
                        help='shared dim after weighted multi-layer ID fusion')

    parser.add_argument('--distill_temp', type=float, default=2.0,
                        help='temperature for continuous->ID distillation')
    parser.add_argument('--distill_weight', type=float, default=0.3,
                        help='weight of distillation loss')
