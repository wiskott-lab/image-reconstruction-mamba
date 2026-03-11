import neptune
import config
import pandas as pd
import torch
import sys
import tools.strings as strings
import tools.neptune_utils as nu
from copy import deepcopy

# Some code to verify checkpoints in neptune workspace
def convert_state_dict(obj):
    return {key: value.cpu() if torch.is_tensor(value) else value for key, value in obj.state_dict().items()}


def generate_checkpoint(model, optimizer, best_loss, val_step, train_step, lr_scheduler=None, loss_scaler=None):
    checkpoint = {
        strings.MODEL_ST_KEY: convert_state_dict(getattr(model, "_orig_mod", model)) if model is not None else None,
        strings.OPTIMIZER_ST_KEY: convert_state_dict(optimizer) if optimizer is not None else None,
        strings.LR_SCHEDULER_ST_KEY: convert_state_dict(lr_scheduler) if lr_scheduler is not None else None,
        strings.LOSS_SCALER_ST_KEY: convert_state_dict(loss_scaler) if loss_scaler is not None else None,
        strings.BEST_LOSS_KEY: best_loss,
        strings.TRAIN_STEP_KEY: train_step,
        strings.VAL_STEP_KEY: val_step
    }
    return checkpoint


def check_checkpoint(checkpoint):  # only checks model
    model_st = checkpoint['model_st']
    for key, tensor in model_st.items():
        if isinstance(tensor, torch.Tensor) and torch.isnan(tensor).any():
            print(f"NaN detected in tensor '{key}'")
            sys.exit()  # Stops the program


def verify_run(run_id):
    print(f'Verifying {run_id}...')
    run = neptune.init_run(with_id=run_id, mode='read-only')
    # neptune_params = get_params(run_id)
    checkpoint = nu.get_checkpoint(run_id)
    verify_checkpoint(run, checkpoint)
    if run['sys/failed'].fetch():
        print('Run marked as failed.')
    run.stop()
    print(f'Finished verification of {run_id}.')


def verify_checkpoint(run, checkpoint):
    train_series_ids = list(run.get_structure()["train"].keys())
    val_series_ids = list(run.get_structure()["val"].keys())
    train_step, val_step = checkpoint['train_step'], checkpoint['val_step']
    for series_id in train_series_ids:
        verify_series(run, 'train/' + series_id, train_step)
    for series_id in val_series_ids:
        verify_series(run, 'val/' + series_id, val_step)


def clean_run(run_id, train_step=None, val_step=None):
    print(f'Cleaning {run_id}...')
    run = neptune.init_run(with_id=run_id)
    checkpoint = nu.get_checkpoint(run_id)
    train_series_ids = list(run.get_structure()["train"].keys())
    val_series_ids = list(run.get_structure()["val"].keys())
    if not train_step:
        train_step = checkpoint['train_step']
    if not val_step:
        val_step = checkpoint['val_step']
    for series_id in train_series_ids:
        clean_series(run, 'train/' + series_id, end_index=train_step + 1)
    for series_id in val_series_ids:
        clean_series(run, 'val/' + series_id, end_index=val_step + 1)
    run['sys/failed'] = False
    run.stop()
    print(f'Finished cleaning {run_id}.')


def set_failed_status(run_id, failed=False):
    run = neptune.init_run(with_id=run_id)
    run['sys/failed'] = failed
    run.stop()


def verify_series(run, series_id, step):
    if step == -1:
        print('No record for ' + series_id)
    else:
        series_df = run[series_id].fetch_values()
        if series_df.shape[0] - 1 != step:
            print(series_id + ': received shape ' + str(series_df.shape[0]) + ' does not equal expected shape '
                  + str(step + 1))
        else:
            print(series_id + ' verified: ' + str(series_df.shape[0]) + '/' + str(step + 1))


def clean_series(run, series_id, end_index):
    series_df = run[series_id].fetch_values()
    series_df.to_pickle(config.TMP_DIR / (run["sys/id"].fetch() + '_' + series_id.replace("/", "_") + '.pkl'),
                        compression='infer', protocol=5, storage_options=None)  # make a local copy of series to
    # prevent data loss. Safety first!
    series_df = series_df.drop(series_df.index[end_index:])
    del run[series_id]
    for index, row in series_df.iterrows():
        run[series_id].append(value=row['value'], step=row['step'],
                              timestamp=row['timestamp'].to_pydatetime().timestamp())
        # need to convert to pydate time because pandas handles timezone conversions differently and neptune backend is
        # using pytime and pandas implicitly casts the timestamps received from server to pandas timestamp


def upload_series(run_id, file, series_id):
    series_df = pd.read_pickle(file)
    run = neptune.init_run(with_id=run_id)
    for index, row in series_df.iterrows():
        run[series_id].append(value=row['value'], step=row['step'], timestamp=row['timestamp'].timestamp())


def clean_and_verify_run(run_id):
    clean_run(run_id)
    verify_run(run_id)



# checkpoint methods
def get_local_cp_filenames(run_path):
    return [f.name for f in run_path.glob("*.cp")]


def get_online_cp_filenames(run):
    return list(run.get_structure()[strings.CHECKPOINT].keys())


def del_online_cp_files(run, cps_to_keep):
    cp_files = get_online_cp_filenames(run)
    ascending_cps = sorted(cp_files, key=str)
    if len(ascending_cps) > cps_to_keep:
        for cp_file in ascending_cps[:-cps_to_keep]:
            del run[strings.CHECKPOINTS][cp_file]




if __name__ == '__main__':
    pass
