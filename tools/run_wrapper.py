import config
import uuid
import numpy as np
import json
import tools.strings as strings


class OfflineRun:

    def __init__(self, run_id=None, data=None, continue_run=False):
        self._data = {} if data is None else data
        self.run_id = run_id
        if continue_run:
            self.load_cfg()
            self.load_series()

    def __getitem__(self, key):
        if key not in self._data:
            self._data[key] = []
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def save(self):
        with open(str(config.RUNS_PATH / self.run_id / 'cfg.json'), "w") as f:
            json.dump(self._data['cfg'], f, indent=4)
        self.save_series()

    def load_cfg(self):
        with open(str(config.RUNS_PATH / self.run_id / 'cfg.json'), "r") as f:
            self._data['cfg'] = json.load(f)
        self.save_series()

    def save_series(self):
        series_keys = [k for k in self._data if k.startswith("series/")]
        for key in series_keys:
            path_elements = key.split('/')
            np.save(str(config.RUNS_PATH / self.run_id / path_elements[0] / path_elements[1] / (path_elements[2] + '.npy')), np.array(self._data[key]))

    def load_series(self):
        series_dir = config.RUNS_PATH / self.run_id / 'series'
        series_sub_dirs = [p.name for p in series_dir.iterdir() if p.is_dir()]
        for series_sub_dir in series_sub_dirs:
            series_paths = [p for p in (series_dir / series_sub_dir).iterdir() if p.is_file() and p.suffix == ".npy"]
            for series_path in series_paths:
                values = np.load(series_path).tolist()
                self['series/' + series_sub_dir + '/' + series_path.stem].extend(values)


    def get_series(self):
        series = []
        for key, value in self._data.items():
            if key.startswith("series/"):
                series.append((key, value))
        return series

    def stop(self):
        self.save()


# class Container:
#
#     def __init__(self, key):
#         super().__init__()
#         self.value = None
#         self.key = key
#
#     def append(self, value):
#         if self.value is None:
#             self.value = []
#         self.value.append(value)
#
#     def extend(self, values):
#         if self.value is None:
#             self.value = []
#         self.value.extend(values)

    #
    # class CPU_Unpickler(pickle.Unpickler):
    #     def find_class(self, module, name):
    #         if module == 'torch.storage' and name == '_load_from_bytes':
    #             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
    #         return super().find_class(module, name)
    #
    #

    # @classmethod
    # def from_run_id(cls, run_id=None):
    #     offline_run = cls(run_id=run_id)
    #     with open(config.RUNS_PATH / run_id / 'cfg.json', "r") as f:
    #         cfg = json.load(f)
    #     offline_run['cfg'] = cfg
    #     OfflineRun.load_series(offline_run)
    #     # OfflineRun.load_files(offline_run)
    #     return offline_run

    #
    # def save_files(self):
    #     files_keys = [k for k in self._data if k.startswith("files/")]
    #     for key in files_keys:
    #         path_elements = key.split('/')
    #         torch.save(f=str(config.RUNS_PATH / self.run_id / path_elements[0] / path_elements[1]), obj=self._data[key])

    # @staticmethod
    # def get_files(run, run_id):
    #     run[strings.CHECKPOINT].download(str(config.RUNS_PATH / run_id / 'files' / strings.CHECKPOINT))
    #     run[strings.MODEL_ST].download(str(config.RUNS_PATH / run_id / 'files' / strings.MODEL_ST))

    # @staticmethod
    # def get_cfg(run, run_id):
    #     with open(str(config.RUNS_PATH / run_id / 'cfg.json'), "w") as f:
    #         json.dump(run['cfg'].fetch(), f, indent=4)

    # @staticmethod
    # def load_files(offline_run):
    #     files_dir = config.RUNS_PATH / offline_run.run_id / 'files'
    #     files_ids = [p.name for p in files_dir.iterdir() if p.is_file() and p.suffix == ".pt"]
    #     for file_id in files_ids:
    #         file = torch.load(files_dir / file_id)
    #         offline_run['files/' + file_id] = file

    # @staticmethod
    # def get_neptune_series(key_prefix, key_suffix, run, run_id):
    #     values = np.array(list(run[key_prefix + '/' + key_suffix].fetch_values()['value']))
    #     np.save(str(config.RUNS_PATH / run_id / 'series' / key_prefix / (key_suffix + '.npy')), values)

    # def sync_to_run_id(self, with_id=None):
    #     run = neptune.init_run(with_id=with_id, capture_stderr=False, capture_stdout=False, mode='async',
    #                            capture_traceback=False, capture_hardware_metrics=False)
    #     for key, value in self._data.items():
    #         if key == 'cfg':
    #             run['cfg'] = value
    #         elif isinstance(value.value, list):
    #             self.sync_list(run, key, value.value)
    #         else:
    #             tmp_path = str(uuid.uuid4())
    #             torch.save(obj=value.value, f=tmp_path)
    #             run[key].upload(tmp_path, wait=True)
    #             os.remove(tmp_path)
    #     run.stop()
    # @staticmethod
    # def sync(offline_run_id, neptune_run_id=None):
    #     offline_run = OfflineRun.from_run_id(offline_run_id)
    #     offline_run.sync_to_run_id(with_id=neptune_run_id)
    #
    # @staticmethod
    # def sync_list(run, key, values):
    #     for value in values:
    #         run[key].append(value)





if __name__ == '__main__':
    a = OfflineRun(run_id='SE-1758', data=None, continue_run=True)
    pass
    # OfflineRun.to_offline_run(with_id='SE-1722')
    # offline_run = OfflineRun.from_run_id(run_id='SE-1722')
    # offline_run.save()


    # offline_run = OfflineRun.from_run_id(path='/Users/rathjjgf/6e577e9d-871f-4fa4-a0a7-1607b9dc1b2b.pkl')
    # offline_run.sync_to_run_id()
    pass
