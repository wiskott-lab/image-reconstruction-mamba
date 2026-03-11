from torch.amp import autocast

import torch
import config
import argparse
from pathlib import Path
import tools.neptune_utils as nu

from tools.dataloader_builder import init_dataloaders
from tools.misc_utils import get_parent_file, tuples_to_strings, parse_cfg
import os
import neptune
from tools import strings
from training.vit_engine import train
from types import SimpleNamespace
from collections import OrderedDict

if __name__ == '__main__':
    os.nice(10)  # Adjusts the process priority by +10
    parser = argparse.ArgumentParser()
    # script params
    parser.add_argument("--cuda_device", "-c", help='cuda_device', type=int, default=0)
    parser.add_argument("--init_run_id", type=str, default=None)
    parser.add_argument("--epochs", "-e", help='number_of_epochs', type=int, default=200000)
    parser.add_argument("--with_id", "-r", help='neptune id run', type=str, default=None)
    parser.add_argument("--neptune_mode", "-n", type=str, default=None)
    parser.add_argument('--job_id', '-j', type=str, default=None)
    # optimisation params
    parser.add_argument("--lr", help='learning rate', type=float, default=0.0003)
    # model args
    parser.add_argument("--use_pos_emb", type=bool, default=True)
    parser.add_argument("--d_model", help='dimensions per token', type=int, default=16)
    parser.add_argument("--n_layers", help='number of layers', type=int, default=8)
    parser.add_argument('--patch_sizes', type=int, nargs='+', default=(4,))
    parser.add_argument('--resize_sizes', type=int, nargs='+', default=(4,))
    parser.add_argument("--pos_emb_type", "-pet", help='pos_emb_type', type=str, default='add')
    # env args
    parser.add_argument("--batch_size", "-bs", help='batch size', type=int, default=64)
    parser.add_argument('--seq_len_range', type=int, nargs=2, default=(1, 128))
    parser.add_argument('--query_len_range', type=int, nargs=2, default=(1, 100))
    parser.add_argument("--intermediate_save", "-is", help='intermediate save', type=int, default=2000)
    parser.add_argument("--init_from", '-ir', help='neptune id run', type=str, default=None)
    parser.add_argument("--clip", help='clip gradient', type=float, default=1.0)

    args = parser.parse_args()

    img_shape = 128
    # img_shape = 28
    # img_shape = 128

    foveation_properties_cfg = {'patch_sizes': tuple(args.patch_sizes), 'resize_sizes': tuple(args.resize_sizes)}
    model_cfg = {'model_name': 'TransformerEncoderDecoder', 'embed_dim': args.d_model, 'num_heads': 8,
                 'num_layers': args.n_layers, 'ff_dim': 4, 'num_channels': 1,
                 'drop_path': 0.05, 'pos_emb_type': args.pos_emb_type,
                 'foveation_properties_cfg': foveation_properties_cfg,
                 'decoder_cfg': {'model_name': 'PatchDecoder', 'in_features': args.d_model,
                                 'out_channels': 1, 'patch_size': args.patch_sizes[-1]}}
    optim_cfg = {'opt': 'Lamb', 'lr': args.lr}
    lr_scheduler_cfg = {'sched': 'cosine', 'warmup_lr': 0.000001, 'min_lr': 0.00001, 'noise_std': 0.1,
                        'noise_pct': 0.4, 'warmup_epochs': 5, 'decay_epochs': 30, 'cooldown_epochs': 10,
                        'patience_epochs': 10, 'decay_rate': 0.1, 'num_epochs': args.epochs}
    loss_scaler_cfg = {'scaler': 'NativeScaler'}

    #
    transform_train_cfg = {'0': {'class_name': 'ToTensor'},
                           '1': {'class_name': 'RandomAffine', 'degrees': 180, 'fill': 1.0, 'scale': (0.5, 1.5),
                                 'translate': (0.4, 0.4)},
                           '2': {'class_name': 'Resize', 'size': (img_shape, img_shape)},
                           '3': {'class_name': 'Normalize', 'mean': (0.5,),
                                 'std': (0.5,)}}
    transform_val_cfg = {'0': {'class_name': 'ToTensor'},
                         '1': {'class_name': 'RandomAffine', 'degrees': 180, 'fill': 1.0, 'scale': (0.5, 1.5),
                               'translate': (0.4, 0.4)},
                         '2': {'class_name': 'Resize', 'size': (img_shape, img_shape)},
                         '3': {'class_name': 'Normalize', 'mean': (0.5,),
                               'std': (0.5,)}}
    dataset_train_cfg = {'dataset_id': 'OMNIGLOT', 'background': 'True'}
    dataset_val_cfg = {'dataset_id': 'OMNIGLOT', 'background': 'False'}

    #
    dataloader_train_cfg = {'num_workers': 4, 'shuffle': True, 'drop_last': True, 'pin_memory': True,
                            'batch_size': args.batch_size}
    dataloader_val_cfg = {'num_workers': 4, 'shuffle': True, 'drop_last': False, 'pin_memory': True,
                          'batch_size': args.batch_size // 4}

    dataloaders_cfg = {strings.TRAIN: {'dataset_cfg': dataset_train_cfg, 'transform_cfg': transform_train_cfg,
                                       'dataloader_cfg': dataloader_train_cfg},
                       strings.VAL: {'dataset_cfg': dataset_val_cfg, 'transform_cfg': transform_val_cfg,
                                     'dataloader_cfg': dataloader_val_cfg}}

    train_args_cfg = {'seq_len_range': tuple(args.seq_len_range), 'seq_len_eval': tuple(args.seq_len_range)[-1],
                 'clip_grad': args.clip, 'intermediate_save': args.intermediate_save,
                 'query_len_range': tuple(args.query_len_range), 'eval_query_len': tuple(args.query_len_range)[-1]}
    cfg = {'scope': get_parent_file(Path(__file__)), strings.MODEL_CFG: model_cfg, strings.OPTIMIZER_CFG: optim_cfg,
           strings.LR_SCHEDULER_CFG: lr_scheduler_cfg, strings.LOSS_SCALER_CFG: loss_scaler_cfg,
           strings.DATALOADERS_CFG: dataloaders_cfg,
           strings.TRAIN_ARGS_CFG: train_args_cfg, 'init_from': args.init_from}

    cfg = tuples_to_strings(cfg)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)

    if args.neptune_mode is not None:
        os.environ["NEPTUNE_MODE"] = args.neptune_mode
    run = nu.init_run(with_id=args.with_id, job_id=args.job_id)

    checkpoint, model_state = None, None
    if args.with_id is not None:
        checkpoint = nu.get_checkpoint(args.with_id)
        cfg = nu.get_cfg(args.with_id)
    elif args.init_from is not None:
        cfg['model_cfg'] = nu.get_cfg(args.with_id)['model_cfg']
        run[strings.CFG] = cfg
        model_state = nu.get_model_state(args.with_id)
    else:
        run[strings.CFG] = cfg

    cfg = parse_cfg(cfg)

    model = nu.init_model(cfg=cfg, checkpoint=checkpoint, state=model_state)
    model.to(config.DEVICE)
    dataloader_train, dataloader_val = init_dataloaders(cfg)
    train_args = cfg[strings.TRAIN_ARGS_CFG]
    optimizer = nu.init_optimizer(cfg=cfg, model=model, checkpoint=checkpoint)
    loss_scaler = nu.init_loss_scaler(cfg=cfg, checkpoint=checkpoint)
    lr_scheduler = None
    best_loss, train_step, val_step = nu.init_train_variables(checkpoint=checkpoint)
    with autocast(enabled=False, device_type='cuda'):
        train(model=model, epochs=args.epochs, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_scaler=loss_scaler,
              run=run, best_loss=best_loss, val_step=val_step, train_step=train_step, dataloader_train=dataloader_train,
              dataloader_val=dataloader_val, **train_args)
