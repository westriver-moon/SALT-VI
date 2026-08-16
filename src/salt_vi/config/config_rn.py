import argparse
import ast
import sys

from .validation import SUPPORTED_JOINT_MODES


def _parse_int_pair(value):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
        raise argparse.ArgumentTypeError(
            f"expected a pair such as '288,144', got {value!r}"
        )
    try:
        return tuple(int(item) for item in parsed)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"image size values must be integers, got {value!r}"
        ) from exc


def build_parser():
    parser = argparse.ArgumentParser()
    # Debug
    parser.add_argument('--DEBUG', default=False, action='store_true')
    parser.add_argument('--DEBUG_DIR', default='/data0/hzy_log/WORK_2024_LOG/Debug_logs', type=str, help='debug dir')

    # Device and DataParallel setting 
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default="0,1,2,3", help="0,1,2,3")
    parser.add_argument('--gpu_id', type=str, default='0',help='using single gpu')
    parser.add_argument('--DataParallel', default=False, action='store_true', help='whether to use DataParallel')

    # training and testing config
    parser.add_argument('--LOG4TEST', default=False, action='store_true') # config file path
    parser.add_argument('--config_select', type=str, default='default', help='config file path') # config file path 
    parser.add_argument(
        '--set',
        dest='config_overrides',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='override any existing YAML key after loading the selected config; may be repeated',
    )
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--test_mode', default='all', type=str, help='all or indoor')
    parser.add_argument('--gall_mode', default='single', type=str, help='single or multi')
    parser.add_argument('--regdb_test_mode', default='t-v', type=str, help='')
    parser.add_argument('--test_model_type', default='Fusion', help='the type of mode for testing["IR","Fusion","Text"]',type=str)
    parser.add_argument('--test_modality', default='Fusion', help='testing retrieval mode ["IR","Fusion","Text"]',type=str)
    parser.add_argument('--training_mode', default='RGB_IR_Text', type=str, help='RGB_IR or RGB_IR_Text')
    parser.add_argument(
        '--joint_mode',
        default='ir_crossfusion',
        choices=SUPPORTED_JOINT_MODES,
        help='supported modes: image_only, ir_crossfusion, uni',
    )



    # LLM augmentation setting 
    parser.add_argument('--llm_aug', default=False, action='store_true', help='whether use llm augmentation in training or not')
    parser.add_argument('--llm_aug_prob', default=0.5, type=float, help='prob for applying llm rephrase')
    # parser.add_argument('--text_guided', default=False, action='store_true', help='whether use text guided and related model to train')
    parser.add_argument('--captioner_name', default='Blip', type=str, help='Use which Captioner type to do the augmentation ["GIT","Blip"]')

    # dataset setting 
    parser.add_argument('--Feat_Filter', default=False, action='store_true')
    parser.add_argument('--training_weight_init', default=None, type=str, help='weight path initialization for training')
    parser.add_argument('--training_weight_init_sha256', default=None, type=str)
    parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu or llcm]')
    parser.add_argument('--sysu_data_path', type=str, default='/data0/hzy_data/SYSU-MM01/')
    parser.add_argument('--regdb_data_path', type=str, default='/data0/hzy_data/RegDB/')
    parser.add_argument('--llcm_data_path', type=str, default='/data0/hzy_data/LLCM/')
    parser.add_argument('--pretrain_path', type=str, default='default', help='pretrained model path')
    parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')
    parser.add_argument('--eval_num_regdb', default=1, type=int) # 1 or 10 for regdb
    parser.add_argument('--batch-size', default=32, type=int, metavar='B', help='training batch size')
    parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
    parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--eval_caption_seed', type=int, default=0)
    parser.add_argument('--pid_num', type=int, default=395)
    parser.add_argument('--num_pos', default=4, type=int,help='num of pos per identity in each modality')
    parser.add_argument('--num_workers', default=8, type=int,help='num of pos per identity in each modality')

 
    ######################## eval and log config during training ########################
    parser.add_argument('--test_model_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='/data0/hzy_log/WORK_2024_LOG/logs/',
                        help='path to save related informations')
    parser.add_argument('--clip_download_root', type=str, default='~/.cache/clip',
                        help='path to cache OpenAI CLIP model weights')
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='-1 for no resuming')
    parser.add_argument('--auto_resume_training_from_lastest_step', action="store_true", default=False)
    parser.add_argument('--eval_epoch', type=int, default=2)
    parser.add_argument('--eval_start_epoch', type=int, default=80)
    parser.add_argument('--checkpoint_epoch', type=int, default=10)
    parser.add_argument('--CAT_EVAL', default=False, action='store_true')

 
    ######################## model general settings ########################
    parser.add_argument('--uni_BN', default=False, action='store_true')
    parser.add_argument('--Fix_Visual', default=False, action='store_true')
    parser.add_argument('--prj_output_dim', type=int, default=2048)
    parser.add_argument("--pooling", default='GEM', type=str, help='["attnpool","GEM"]')
    parser.add_argument("--pretrain_choice", default='RN50_ORI',help='ViT-B/16,RN50,RN50_ORI,PMT_VIT') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.07, help="initial contrastive temperature; must be finite and > 0")
    parser.add_argument("--freeze_text_in_image_only", default=False, action='store_true')

    ######################## cross transfomer setting ########################
    parser.add_argument("--cmt_depth", type=int, default=1, help="cross modal transformer self attn layers")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")

    ######################## loss settings ########################
    # parser.add_argument("--uni_train", default=False, action='store_true', help="whether use uni_modal training")
    parser.add_argument("--loss_names", default='wrt,id', help="which loss to use ['wrt', 'id', 'uni_reid', 'Fusion_Regular']")
    parser.add_argument("--cmm_loss_weight", type=float, default=1.0, help="cross modal matching loss (tcmpm, cmpm, infonce...) weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    parser.add_argument("--wrt_loss_weight", type=float, default=1.0, help="itc loss weight")
    
    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=_parse_int_pair, default=(288, 144))
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--pmt_pretrained", type=str, default=None)
    parser.add_argument("--pmt_embed_dim", type=int, default=768)
    parser.add_argument("--pmt_patch_size", type=ast.literal_eval, default=(16, 16))
    parser.add_argument("--pmt_stride_size", type=ast.literal_eval, default=(12, 12))
    parser.add_argument("--pmt_depth", type=int, default=12)
    parser.add_argument("--pmt_num_heads", type=int, default=12)
    parser.add_argument("--pmt_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--pmt_dropout", type=float, default=0.03)
    parser.add_argument("--pmt_attention_dropout", type=float, default=0.0)
    parser.add_argument(
        "--pmt_attention_backend",
        choices=("manual", "sdpa", "flash"),
        default="manual",
    )
    parser.add_argument("--pmt_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--pmt_patch_embed", type=ast.literal_eval, default=None)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")

    parser.add_argument("--lr", type=float, default=None, help="base learning rate for SGD; defaults to lr_visual")

    parser.add_argument("--lr_txt", type=float, default=1e-5) # 如果使用文字模态的transformer，需要设置一个较小的学习率
    parser.add_argument("--text_weight_decay", type=float, default=4e-5) 
    parser.add_argument("--text_bias_lr_factor", type=float, default=2.) 
    parser.add_argument("--text_weight_decay_bias", type=float, default=0.) # for text 4e-5
    parser.add_argument("--lr_visual", type=float, default=0.00035) # 两个gpu就乘以二 7e-4 # RegDB 0.001
    parser.add_argument("--visual_weight_decay", type=float, default=5e-4)
    parser.add_argument("--visual_bias_lr_factor", type=float, default=1.) 
    parser.add_argument("--visual_weight_decay_bias", type=float, default=5e-4)

    parser.add_argument("--classifier_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--alpha", type=float, default=0.9, help='for adam and adamW')
    parser.add_argument("--beta", type=float, default=0.999, help='for adam and adamW')
    
    ######################## scheduler ########################
    parser.add_argument('--total_train_epoch', type=int, default=120) # 100
    parser.add_argument("--milestones", type=int, nargs='+', default=(40, 60, 100)) # (40,60)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="step")
    parser.add_argument("--target_lr", type=float, default=0.0)
    parser.add_argument("--power", type=float, default=0.9) # 1.0

    ######################## multi-modality model settings ########################
    parser.add_argument(
        "--fusion_way",
        default="add",
        help=(
            "[add, norm_add, parameter_add, adaptive_add, cross_attention, "
            "attention_rgb_text, attention_ir_text, global_attention, "
            "global_attention_rgb_text, global_attention_ir_text]"
        ),
    )
    parser.add_argument("--pa", type=float, default=0.1, help="parameter add for fusion")
    
    return parser


def get_args(argv=None):
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    config = parser.parse_args(raw_args)
    explicit = set()
    for token in raw_args:
        if not token.startswith('-'):
            continue
        option = token.split('=', 1)[0]
        action = parser._option_string_actions.get(option)
        if action is not None:
            explicit.add(action.dest)
    config._explicit_cli_destinations = explicit
    return config
