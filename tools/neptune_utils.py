import os
import torch
import config
import neptune
import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import NativeScaler
import timm.utils.cuda
from tools.model_builder import create_model
from tools.run_wrapper import OfflineRun
import tools.strings as strings
import json
import numpy as np
import uuid

from tools.misc_utils import parse_cfg
from datetime import datetime


def upload_checkpoint(checkpoint, run, intermediate_save=None):
    if intermediate_save is not None:
        val_step = checkpoint[strings.VAL_STEP_KEY]
        if val_step % intermediate_save == 0:
            filename = (strings.CHECKPOINT_KEY + '_' + str(val_step).zfill(7) + '.pt')
            path = config.RUNS_PATH / get_run_id(run) / 'files' / filename
            torch.save(checkpoint, f=path)
            if not isinstance(run, OfflineRun):
                run['files/' + filename].upload(str(path))
    upload_file(checkpoint, strings.CHECKPOINT_FILENAME, run)
    if isinstance(run, OfflineRun):
        run.save()
        update_meta_file(run_id=get_run_id(run))
    else:
        update_meta_file(run_id=get_run_id(run), sync_status='neptune')


def upload_model_state(model_state, run):
    upload_file(model_state, strings.MODEL_ST_FILENAME, run)


def upload_file(obj, filename, run):
    path = config.RUNS_PATH / get_run_id(run) / 'files' / filename
    torch.save(obj, f=path)
    if os.getenv('NEPTUNE_MODE') != 'offline':
        run['files/' + filename].upload(str(path))


def init_run(with_id=None, monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False,
             capture_hardware_metrics=False, job_id=None, neptune_mode=None, *args, **kwargs):
    if neptune_mode == 'offline' or os.getenv('NEPTUNE_MODE') == 'offline':
        if with_id is not None:
            run = OfflineRun(run_id=with_id, continue_run=True)
        else:
            with_id = "offline_" + str(uuid.uuid4())
            run = OfflineRun(run_id=with_id)
        make_dirs(run_id=with_id)
    else:
        run = neptune.init_run(with_id=with_id, monitoring_namespace=monitoring_namespace,
                               capture_stdout=capture_stdout, capture_stderr=capture_stderr,
                               capture_hardware_metrics=capture_hardware_metrics, *args, **kwargs)
        with_id = get_run_id(run)
        make_dirs(run_id=with_id)
        update_meta_file(run_id=with_id, sync_status='neptune')
    return run


def make_dirs(run_id):
    (config.RUNS_PATH / run_id).mkdir(exist_ok=True, parents=True)
    (config.RUNS_PATH / run_id / 'files').mkdir(exist_ok=True, parents=True)
    (config.RUNS_PATH / run_id / 'series').mkdir(exist_ok=True, parents=True)
    (config.RUNS_PATH / run_id / 'series' / 'val').mkdir(exist_ok=True, parents=True)
    (config.RUNS_PATH / run_id / 'series' / 'train').mkdir(exist_ok=True, parents=True)


def create_loss_scaler(scaler, *args, **kwargs):
    return getattr(timm.utils.cuda, scaler)(*args, **kwargs)


def get_run_id(run):
    if isinstance(run, OfflineRun):
        return run.run_id
    else:
        return run["sys/id"].fetch()


def init_model(cfg, checkpoint=None, state=None):
    model = create_model(**cfg[strings.MODEL_CFG])
    if checkpoint is not None:
        model_state = checkpoint['model_st']
        model_state = {k.replace('_orig_mod.', ''): v for k, v in model_state.items()}  #
        model.load_state_dict(model_state)
    elif state is not None:
        model.load_state_dict(state)
    model.to(config.DEVICE)
    return model


def init_optimizer(cfg, model, checkpoint=None, state=None):
    optimizer = create_optimizer_v2(model_or_params=model, **cfg[strings.OPTIMIZER_CFG])
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint[strings.OPTIMIZER_ST_KEY])
    elif state is not None:
        optimizer.load_state_dict(state)
    return optimizer


def init_lr_scheduler(cfg, optimizer, checkpoint=None, state=None):
    lr_scheduler, _ = create_scheduler_v2(optimizer=optimizer, **cfg[strings.LR_SCHEDULER_CFG])
    if checkpoint is not None:
        lr_scheduler.load_state_dict(checkpoint[strings.LR_SCHEDULER_ST_KEY])
    elif state is not None:
        lr_scheduler.load_state_dict(state)
    return lr_scheduler


def init_loss_scaler(cfg, checkpoint=None, state=None):
    loss_scaler = create_loss_scaler(**cfg[strings.LOSS_SCALER_CFG])
    if checkpoint is not None:
        loss_scaler.load_state_dict(checkpoint[strings.LOSS_SCALER_ST_KEY])
    elif state is not None:
        loss_scaler.load_state_dict(state)
    return loss_scaler


def init_train_variables(checkpoint=None):
    best_loss, train_step, val_step = None, -1, 0
    if checkpoint is not None:
        best_loss, train_step, val_step = checkpoint[strings.BEST_LOSS_KEY], checkpoint[strings.TRAIN_STEP_KEY], checkpoint[
            strings.VAL_STEP_KEY],
    return best_loss, train_step, val_step


def get_cfg(run_id, update=False, parse=True, project=None):
    make_dirs(run_id)
    path = config.RUNS_PATH / run_id / 'cfg.json'
    if update or not path.exists():
        run = neptune.init_run(with_id=run_id, mode='read-only', project=project)
        with open(str(path), "w") as f:
            json.dump(run['cfg'].fetch(), f, indent=4)
        run.stop()
    with open(path, "r") as f:
        cfg = json.load(f)
        if parse:
            cfg = parse_cfg(cfg)
    return cfg


def get_checkpoint(run_id, update=False):
    return get_file(run_id, 'checkpoint.pt', update)


def get_model_state(run_id, update=False, project=None, at_cp=None):
    if at_cp is not None:
        return get_file(run_id=run_id, load=True, filename='checkpoint_' + str(at_cp).zfill(7) + '.pt')[strings.MODEL_ST_KEY]
    return get_file(run_id, 'model_st.pt', update, project=project)


def get_file(run_id, filename, update=False, load=True, project=None):
    path = config.RUNS_PATH / run_id / 'files' / filename
    if update or not path.exists():
        run = neptune.init_run(with_id=run_id, mode='read-only', project=project)
        run['files/' + filename].download(str(path))
        run.stop()
    if load:
        return torch.load(path, map_location=torch.device('cpu'))
    return None

def download_files(run_id, project=None):
    run = neptune.init_run(with_id=run_id, mode='read-only', project=project)
    filenames = list(run.get_structure()['files'].keys())
    for filename in filenames:
        run['files/' + filename].download(str(config.RUNS_PATH / run_id / 'files' / filename))
    run.stop()


def get_series(key_prefix, key_suffix, run_id, update=False, load=True):
    path = config.RUNS_PATH / run_id / 'series' / key_prefix / (key_suffix + '.npy')
    if update or not path.exists():
        run = neptune.init_run(with_id=run_id, mode='read-only')
        values = np.array(list(run['series/' + key_prefix + '/' + key_suffix].fetch_values()['value']))
        np.save(str(path), values)
        run.stop()
    if load:
        return np.load(path)
    return None


def download_all_series(run_id):
    run = neptune.init_run(with_id=run_id, mode='read-only')
    train_series_ids = list(run.get_structure()["series"]["train"].keys())
    val_series_ids = list(run.get_structure()["series"]["val"].keys())
    run.stop()
    for series_id in val_series_ids:
        get_series(key_prefix='val', key_suffix=series_id, run_id=run_id, load=False)
    for series_id in train_series_ids:
        get_series(key_prefix='train', key_suffix=series_id, run_id=run_id, load=False)


def to_offline_fov_6(exclude=()):
    df = neptune.init_project(mode="read-only").fetch_runs_table().to_pandas()
    sys_ids = df.query(
        "`cfg/model_cfg/model_name` == 'FoveatedMamba6' and `cfg/dataloaders_cfg/train/dataset_cfg/dataset_id` == 'OMNIGLOT'"
        " and `cfg/model_cfg/mamba_cfg/hidden_size` == 16")["sys/id"].to_list()

    offline_runs = [
        p.name for p in config.RUNS_PATH.iterdir()
        if p.is_dir() and p.name.startswith("SAUR-")
    ]
    for run_id in sys_ids:
        if not run_id in exclude and not run_id in offline_runs :
            to_offline_run([run_id])


def to_offline_trans(exclude=()):
    df = neptune.init_project(mode="read-only").fetch_runs_table().to_pandas()
    sys_ids = df.query(
        "`cfg/model_cfg/model_name` == 'TransformerEncoderDecoder' "
        "and `cfg/dataloaders_cfg/train/dataset_cfg/dataset_id` == 'OMNIGLOT'")["sys/id"].to_list()

    offline_runs = [
        p.name for p in config.RUNS_PATH.iterdir()
        if p.is_dir() and p.name.startswith("SAUR-")
    ]
    for run_id in sys_ids:
        if not run_id in exclude and not run_id in offline_runs :
            to_offline_run([run_id])


def to_offline_run(with_ids):
    for with_id in with_ids:
        make_dirs(with_id)
        download_files(with_id)
        get_cfg(with_id)
        download_all_series(with_id)
        make_meta_file(with_id)
        # get_checkpoint(with_id)
        # get_model_state(with_id)

def sync_offline_run(with_id: str):
    print(f"Syncing run {with_id}...")
    meta_dict = load_meta_file(with_id)
    if meta_dict is None:
        update_meta_file(with_id)
        meta_dict = load_meta_file(with_id)
    if meta_dict['status'] == 'synced':
        print(f"Run {with_id} already synced.")
    elif meta_dict['status'] == 'synced':
        print(f"Run {with_id} already synced.")
    else:
        if with_id.startswith('offline_'):
            neptune_run =  neptune.init_run(monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False, mode='async')
        else:
            neptune_run = neptune.init_run(with_id=with_id, monitoring_namespace='monitoring', capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False, mode='async')
        neptune_run['cfg'] = get_cfg(with_id, parse=False)
        sync_all_series(neptune_run=neptune_run, with_id=with_id)
        sync_files(with_id=with_id, neptune_run=neptune_run)
        neptune_run_id = get_run_id(neptune_run)
        neptune_run.stop()
        if with_id.startswith('offline_'):
            (config.RUNS_PATH / with_id).rename(config.RUNS_PATH / neptune_run_id)
        make_meta_file(neptune_run_id)
        print(f"Finished syncing run {with_id} to {neptune_run_id}.")


def sync_files(with_id, neptune_run):
    neptune_filenames = list(neptune_run.get_structure().get('files', {}).keys())
    pt_files = [p.name for p in (config.RUNS_PATH / with_id / 'files').glob("*.pt")]
    for file in pt_files:
        if file == strings.CHECKPOINT_FILENAME or file == strings.MODEL_ST_FILENAME or not file in neptune_filenames:
            neptune_run['files/' + file].upload(str(config.RUNS_PATH / with_id / 'files' / file))


def sync_series_family(neptune_run, series_family_key, with_id, step):
    series_keys = [p.name for p in (config.RUNS_PATH / with_id / 'series' / series_family_key).glob("*.npy")]
    for series_key in series_keys:
        series_values = np.load(config.RUNS_PATH / with_id / 'series' / series_family_key / series_key)[step:].tolist()
        if len(series_values) == 0:
            print(f"No new values to sync for {series_family_key}/{series_key}")
        else:
            neptune_run['series/' + series_family_key + '/' + series_key[:-4]].extend(series_values)


def sync_all_series(with_id, neptune_run):
    meta_file = load_meta_file(with_id)
    series_family_keys = [p.name for p in (config.RUNS_PATH / with_id / 'series').iterdir() if p.is_dir()]
    for series_family_key in series_family_keys:
        sync_series_family(neptune_run=neptune_run, series_family_key=series_family_key,
                           step=meta_file['last_' + series_family_key + '_step'], with_id=with_id)


def init_model_from_neptune(run_id, update=False, project=None, at_cp=None):
    make_dirs(run_id=run_id)
    cfg = get_cfg(run_id, update, project=project)
    model_state = get_model_state(run_id, project=project, at_cp=at_cp, update=update)
    model = init_model(cfg=cfg, state=model_state)
    return model


def sync_offline_runs():
    run_ids = [p.name for p in config.RUNS_PATH.iterdir() if p.is_dir()]
    for run_id in run_ids:
        sync_offline_run(run_id)


def make_meta_file(run_id):
    checkpoint = get_checkpoint(run_id=run_id, update=False)
    meta_dict = {'status': 'synced', 'last_synced': str(datetime.now()),
                 'last_val_step': checkpoint[strings.VAL_STEP_KEY] ,
                 'last_train_step': checkpoint[strings.TRAIN_STEP_KEY]}
    write_meta_file(run_id, meta_dict)


def load_meta_file(run_id):
    path = config.RUNS_PATH / run_id / "META.json"
    if not (config.RUNS_PATH / run_id / "META.json").exists():
        return None
    with open(path) as f:
        meta_dict = json.load(f)
    return meta_dict


def write_meta_file(run_id, meta_dict):
    path = config.RUNS_PATH / run_id / "META.json"
    with open(path, "w") as f:
        json.dump(meta_dict, f)



def get_int_cps_ids(run_id):
    run = neptune.init_run(with_id=run_id, mode='read-only')
    files = list(run.get_structure()['files'].keys())
    int_cps = []
    for key in files:
        if key.startswith('checkpoint_') and key != 'checkpoint_0000000.pt':
            int_cps.append(key)
    return int_cps


def update_meta_file(run_id, sync_status='unsynced'):
    path = config.RUNS_PATH / run_id / "META.json"
    if not path.exists():
        meta_dict = {'status': sync_status, 'last_synced': None, 'last_val_step': 0,'last_train_step': 0}
    else:
        meta_dict = load_meta_file(run_id)
        meta_dict['status'] = sync_status
    write_meta_file(run_id, meta_dict)


def clean_run(run_id):
    run = init_run(with_id=run_id, mode='async')
    cp_ids = get_int_cps_ids(run_id)
    sorted_run_ids = sorted(cp_ids)
    checkpoint = get_file(run_id=run_id, filename=sorted_run_ids[-1], load=True)
    run['files/checkpoint.pt'].upload(str(config.RUNS_PATH / run_id / 'files' / sorted_run_ids[-1]))
    download_all_series(run_id)
    val_step, train_step = checkpoint[strings.VAL_STEP_KEY], checkpoint[strings.TRAIN_STEP_KEY]
    train_keys = [p.name for p in (config.RUNS_PATH / run_id / 'series' / 'train').glob("*.npy")]
    val_keys = [p.name for p in (config.RUNS_PATH / run_id / 'series' / 'val').glob("*.npy")]
    for series_key in train_keys:
        series_values = np.load(config.RUNS_PATH / run_id / 'series' / 'train' / series_key)[:train_step]
        np.save(str(config.RUNS_PATH / run_id / 'series' / 'train' / series_key), series_values)
        del run['series/train/' + series_key[:-4]]
        run['series/train/' + series_key[:-4]].extend(series_values.tolist())
    for series_key in val_keys:
        series_values = np.load(config.RUNS_PATH / run_id / 'series' / 'val' / series_key)[:val_step]
        np.save(str(config.RUNS_PATH / run_id / 'series' / 'val' / series_key), series_values)
        del run['series/val/' + series_key[:-4]]
        run['series/val/' + series_key[:-4]].extend(series_values.tolist())
    run.stop()


if __name__ == '__main__':
    pass