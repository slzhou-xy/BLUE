import argparse
import yaml
import os
import pandas as pd
import numpy as np
import pyproj
from rich.progress import Progress, BarColumn, SpinnerColumn, TimeRemainingColumn, TimeElapsedColumn

from utils import Config, read_pickle, save_pickle
from time_features import time_features


class DataPreprocess:
    def __init__(self, args):
        self.root = args.root
        self.dataset_name = args.dataset_name
        self.traj_file = os.path.join(self.root, self.dataset_name, args.traj_file)
        self.data_save_dir = os.path.join(self.root, self.dataset_name)
        self.max_patch_len_s2 = 0
        self.max_patch_len_s3 = 0

    def _get_points_feature(self, gps_seq, time_seq, geodesic):
        assert len(gps_seq) == len(time_seq)

        lons = gps_seq[:, 0]
        lats = gps_seq[:, 1]

        az1, az2, dist = geodesic.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
        fwd_az = np.concatenate([az1, np.array([0])])
        bk_az = np.concatenate([np.array([0]), az2])

        fwd_dist = np.concatenate([dist, np.array([0])])
        bk_dist = np.concatenate([np.array([0]), dist])

        return fwd_dist, bk_dist, fwd_az, bk_az

    def _get_multi_scale_traj(self, gps_seq, scale=3):
        gps_seq = np.round(gps_seq, decimals=scale)

        diff_mask = np.any(gps_seq[1:] != gps_seq[:-1], axis=1)
        start_index = np.where(diff_mask)[0] + 1
        patch_lens = np.diff(np.concatenate(([0], start_index, [len(gps_seq)])))

        unique_index = np.insert(diff_mask, 0, True)  # 第一个元素总是唯一的
        raw_index = np.where(unique_index)[0]

        gps_seq = gps_seq[raw_index]

        return gps_seq, patch_lens

    def _get_features(self, traj_df, desc):
        traj_df = traj_df[['traj_id', 'user_id', 'class_type', 'traj_len', 'time_seq', 'gps_seq']]
        geodesic = pyproj.Geod(ellps='WGS84')

        total_data = []

        with Progress(
                "[progress.description]{task.description}({task.completed}/{task.total})",
                SpinnerColumn(finished_text="[green]✔"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.2f}%",
                "[yellow]⏱",
                TimeElapsedColumn(),
                "[cyan]⏳",
                TimeRemainingColumn()) as progress:

            description = f"[red]{desc} preprocess"
            task = progress.add_task(description, total=traj_df.shape[0])

            for i, row in traj_df.iterrows():
                if i == traj_df.shape[0]:
                    description = "[green]Finished"
                time_seq = np.array(row['time_seq'][:self.max_len])
                gps_seq = np.vstack(row['gps_seq'][:self.max_len])

                fwd_dist, bk_dist, fwd_az, bk_az = self._get_points_feature(gps_seq, time_seq, geodesic)
                gps_seq_s3, patch_lens_s3 = self._get_multi_scale_traj(gps_seq, scale=3)
                gps_seq_s2, patch_lens_s2 = self._get_multi_scale_traj(gps_seq_s3, scale=2)

                row_data = {
                    'traj_id': row.traj_id,
                    'user_id': row.user_id,
                    'class_type': row.class_type,
                    'traj_len_s5': self.max_len if row.traj_len >= self.max_len else row.traj_len,
                    'gps_seq': gps_seq,
                    'time_seq': time_seq,
                    'time_fea': time_features(pd.to_datetime(time_seq, unit='s'), freq='s').T,
                    'fwd_dist': fwd_dist,
                    'bk_dist': bk_dist,
                    'fwd_az': fwd_az,
                    'bk_az': bk_az,
                    'traj_len_s2': gps_seq_s2.shape[0],
                    'patch_len_s2': patch_lens_s2,
                    'traj_len_s3': gps_seq_s3.shape[0],
                    'patch_len_s3': patch_lens_s3
                }
                total_data.append(row_data)

                progress.update(task, completed=i, description=description)

                self.max_patch_len_s2 = max(self.max_patch_len_s2, max(patch_lens_s2))
                self.max_patch_len_s3 = max(self.max_patch_len_s3, max(patch_lens_s3))

        return total_data

    def run_split(self):
        train_traj_path = os.path.join(self.data_save_dir, 'traj_train.pkl')
        eval_traj_path = os.path.join(self.data_save_dir, 'traj_eval.pkl')
        test_traj_path = os.path.join(self.data_save_dir, 'traj_test.pkl')
        if os.path.exists(eval_traj_path):
            train_traj_data = read_pickle(train_traj_path)
            eval_traj_data = read_pickle(eval_traj_path)
            test_traj_data = read_pickle(test_traj_path)
            return train_traj_data, eval_traj_data, test_traj_data

        traj_df = pd.read_parquet(self.traj_file)
        if 'call_type' in traj_df.columns:
            traj_df = traj_df.rename(columns={'call_type': 'class_type'})
        elif 'flag' in traj_df.columns:
            traj_df = traj_df.rename(columns={'flag': 'class_type'})
        elif 'user_id' in traj_df.columns:
            traj_df = traj_df.rename(columns={'user_id': 'class_type'})
            traj_df['user_id'] = traj_df['class_type']
        else:
            raise NotImplementedError('No class!')

        if not os.path.exists(os.path.join(self.data_save_dir, 'train_index.npy')):
            index = np.arange(traj_df.shape[0])
            index = np.random.permutation(index)
            if self.dataset_name == 'rome':
                train_index = index[:int(0.8 * index.shape[0])]
                eval_index = index[int(0.8 * index.shape[0]): int(0.9 * index.shape[0])]
                test_index = index[int(0.9 * index.shape[0]):]
            else:
                train_index = index[:int(0.6 * index.shape[0])]
                eval_index = index[int(0.6 * index.shape[0]): int(0.8 * index.shape[0])]
                test_index = index[int(0.8 * index.shape[0]):]
            np.save(os.path.join(self.data_save_dir, 'train_index.npy'), train_index)
            np.save(os.path.join(self.data_save_dir, 'eval_index.npy'), eval_index)
            np.save(os.path.join(self.data_save_dir, 'test_index.npy'), test_index)
        else:
            train_index = np.load(os.path.join(self.data_save_dir, 'train_index.npy'))
            eval_index = np.load(os.path.join(self.data_save_dir, 'eval_index.npy'))
            test_index = np.load(os.path.join(self.data_save_dir, 'test_index.npy'))

        train_traj_df = traj_df.iloc[train_index]
        train_traj_df.reset_index(drop=True, inplace=True)
        train_traj_data = self._get_features(train_traj_df, 'Train')
        save_pickle(train_traj_data, train_traj_path)

        eval_traj_df = traj_df.iloc[eval_index]
        eval_traj_df.reset_index(drop=True, inplace=True)
        eval_traj_data = self._get_features(eval_traj_df, 'Eval ')
        save_pickle(eval_traj_data, eval_traj_path)

        test_traj_df = traj_df.iloc[test_index]
        test_traj_df.reset_index(drop=True, inplace=True)
        test_traj_data = self._get_features(test_traj_df, 'Test ')
        save_pickle(test_traj_data, test_traj_path)

        print(f'max_patch_len_s2: {self.max_patch_len_s2}, max_patch_len_s2: {self.max_patch_len_s3}')

        return train_traj_data, eval_traj_data, test_traj_data

    def run(self, save_file, raw_file):
        if os.path.exists(save_file):
            dict_data = read_pickle(save_file)
            return dict_data

        traj_df = pd.read_parquet(raw_file)
        if 'call_type' in traj_df.columns:
            traj_df = traj_df.rename(columns={'call_type': 'class_type'})
        elif 'flag' in traj_df.columns:
            traj_df = traj_df.rename(columns={'flag': 'class_type'})
        elif 'user_id' in traj_df.columns:
            traj_df = traj_df.rename(columns={'user_id': 'class_type'})
            traj_df['user_id'] = traj_df['class_type']
        else:
            raise NotImplementedError('No class!')
        dict_data = self._get_features(traj_df, 'Sim')
        save_pickle(dict_data, save_file)

        return dict_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='chengdu')
    opt = parser.parse_args()

    yaml_path = opt.config

    with open('./config/' + yaml_path + '.yaml', 'r') as f:
        opt = yaml.full_load(f)
    opt = Config(opt)
    data_preprocess = DataPreprocess(opt)
    train_df, eval_df, test_df = data_preprocess.run_split()
