import os
import time
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from preprocess import DataPreprocess


class TrajDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def collate_fn_pretrain(batch_data_list, center, mbr):
    batch_len = len(batch_data_list)
    traj_len_s5 = [data['traj_len_s5'] for data in batch_data_list]
    max_len_s5 = max(traj_len_s5)

    traj_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    mask_y = torch.zeros((batch_len, max_len_s5), dtype=torch.float32)
    time_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)

    patch_len_s2 = []
    patch_len_s3 = []

    traj_len_s2 = []
    traj_len_s3 = []


    for k, data in enumerate(batch_data_list):
        traj_len_s5_k = data['traj_len_s5']
        gps_seq = torch.as_tensor(data['gps_seq'], dtype=torch.float32)
        gps_seq[:, 0] = (gps_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        gps_seq[:, 1] = (gps_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])


        time_fea = torch.as_tensor(data['time_fea'], dtype=torch.float32)
        fwd_dist = torch.as_tensor(data['fwd_dist'] / 1000, dtype=torch.float32)
        bk_dist = torch.as_tensor(data['bk_dist'] / 1000, dtype=torch.float32)
        fwd_az = torch.as_tensor(data['fwd_az'] / 180, dtype=torch.float32)
        bk_az = torch.as_tensor(data['bk_az'] / 180, dtype=torch.float32)

        traj_k = torch.cat([gps_seq, fwd_dist.unsqueeze(-1), bk_dist.unsqueeze(-1), fwd_az.unsqueeze(-1), bk_az.unsqueeze(-1)], axis=1)
        traj_x[k, :traj_len_s5_k] = traj_k

        mask_y[k, :traj_len_s5_k] = 1.0

        time_x[k, :traj_len_s5_k] = time_fea

        patch_len_s2_k = data['patch_len_s2']
        traj_len_s2.append(len(patch_len_s2_k))

        patch_len_s3_k = data['patch_len_s3']
        traj_len_s3.append(len(patch_len_s3_k))

        patch_len_s2.extend(patch_len_s2_k)
        patch_len_s3.extend(patch_len_s3_k)

    data = {
        'x': traj_x,
        'mask_y': mask_y,
        'time_x': time_x,
        'traj_len_s5': torch.as_tensor(traj_len_s5, dtype=torch.long),
        'patch_len_s3': torch.as_tensor(patch_len_s3, dtype=torch.long),
        'traj_len_s3': torch.as_tensor(traj_len_s3, dtype=torch.long),
        'patch_len_s2': torch.as_tensor(patch_len_s2, dtype=torch.long),
        'traj_len_s2': torch.as_tensor(traj_len_s2, dtype=torch.long),
    }

    return data


def collate_fn_inference(batch_data_list, center, mbr):
    batch_len = len(batch_data_list)
    traj_len_s5 = [data['traj_len_s5'] for data in batch_data_list]
    max_len_s5 = max(traj_len_s5)

    traj_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    time_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)

    patch_len_s2 = []
    patch_len_s3 = []

    traj_len_s2 = []
    traj_len_s3 = []

    for k, data in enumerate(batch_data_list):
        traj_len_s5_k = data['traj_len_s5']
        gps_seq = torch.as_tensor(data['gps_seq'], dtype=torch.float32)
        gps_seq[:, 0] = (gps_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        gps_seq[:, 1] = (gps_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])


        time_fea = torch.as_tensor(data['time_fea'], dtype=torch.float32)
        fwd_dist = torch.as_tensor(data['fwd_dist'] / 1000, dtype=torch.float32)
        bk_dist = torch.as_tensor(data['bk_dist'] / 1000, dtype=torch.float32)
        fwd_az = torch.as_tensor(data['fwd_az'] / 180, dtype=torch.float32)
        bk_az = torch.as_tensor(data['bk_az'] / 180, dtype=torch.float32)

        traj_k = torch.cat([gps_seq, fwd_dist.unsqueeze(-1), bk_dist.unsqueeze(-1), fwd_az.unsqueeze(-1), bk_az.unsqueeze(-1)], axis=1)
        traj_x[k, :traj_len_s5_k] = traj_k
        time_x[k, :traj_len_s5_k] = time_fea

        patch_len_s2_k = data['patch_len_s2']
        traj_len_s2.append(len(patch_len_s2_k))

        patch_len_s3_k = data['patch_len_s3']
        traj_len_s3.append(len(patch_len_s3_k))

        patch_len_s2.extend(patch_len_s2_k)
        patch_len_s3.extend(patch_len_s3_k)

    data = {
        'x': traj_x,
        'time_x': time_x,
        'traj_len_s5': torch.as_tensor(traj_len_s5, dtype=torch.long),
        'patch_len_s3': torch.as_tensor(patch_len_s3, dtype=torch.long),
        'traj_len_s3': torch.as_tensor(traj_len_s3, dtype=torch.long),
        'patch_len_s2': torch.as_tensor(patch_len_s2, dtype=torch.long),
        'traj_len_s2': torch.as_tensor(traj_len_s2, dtype=torch.long),
    }

    return data


def collate_fn_cls(batch_data_list, center, mbr):
    batch_len = len(batch_data_list)
    traj_len_s5 = [data['traj_len_s5'] for data in batch_data_list]
    max_len_s5 = max(traj_len_s5)

    traj_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    time_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)

    patch_len_s2 = []
    patch_len_s3 = []

    traj_len_s2 = []
    traj_len_s3 = []

    labels = []


    for k, data in enumerate(batch_data_list):
        traj_len_s5_k = data['traj_len_s5']
        gps_seq = torch.as_tensor(data['gps_seq'], dtype=torch.float32)
        gps_seq[:, 0] = (gps_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        gps_seq[:, 1] = (gps_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])


        time_fea = torch.as_tensor(data['time_fea'], dtype=torch.float32)
        fwd_dist = torch.as_tensor(data['fwd_dist'] / 1000, dtype=torch.float32)
        bk_dist = torch.as_tensor(data['bk_dist'] / 1000, dtype=torch.float32)
        fwd_az = torch.as_tensor(data['fwd_az'] / 180, dtype=torch.float32)
        bk_az = torch.as_tensor(data['bk_az'] / 180, dtype=torch.float32)

        traj_k = torch.cat([gps_seq, fwd_dist.unsqueeze(-1), bk_dist.unsqueeze(-1), fwd_az.unsqueeze(-1), bk_az.unsqueeze(-1)], axis=1)
        traj_x[k, :traj_len_s5_k] = traj_k
        time_x[k, :traj_len_s5_k] = time_fea

        patch_len_s2_k = data['patch_len_s2']
        traj_len_s2.append(len(patch_len_s2_k))

        patch_len_s3_k = data['patch_len_s3']
        traj_len_s3.append(len(patch_len_s3_k))

        patch_len_s2.extend(patch_len_s2_k)
        patch_len_s3.extend(patch_len_s3_k)

        labels.append(data['class_type'])

    data = {
        'x': traj_x,
        'time_x': time_x,
        'traj_len_s5': torch.as_tensor(traj_len_s5, dtype=torch.long),
        'patch_len_s3': torch.as_tensor(patch_len_s3, dtype=torch.long),
        'traj_len_s3': torch.as_tensor(traj_len_s3, dtype=torch.long),
        'patch_len_s2': torch.as_tensor(patch_len_s2, dtype=torch.long),
        'traj_len_s2': torch.as_tensor(traj_len_s2, dtype=torch.long),
    }

    return data, torch.as_tensor(labels, dtype=torch.long)


def collate_fn_tte(batch_data_list, center, mbr):
    batch_len = len(batch_data_list)
    traj_len_s5 = [data['traj_len_s5'] for data in batch_data_list]
    max_len_s5 = max(traj_len_s5)

    traj_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)
    time_x = torch.zeros((batch_len, max_len_s5, 6), dtype=torch.float32)

    patch_len_s2 = []
    patch_len_s3 = []

    traj_len_s2 = []
    traj_len_s3 = []

    labels = []


    for k, data in enumerate(batch_data_list):
        traj_len_s5_k = data['traj_len_s5']
        gps_seq = torch.as_tensor(data['gps_seq'], dtype=torch.float32)
        gps_seq[:, 0] = (gps_seq[:, 0] - mbr['min_lon']) / (mbr['max_lon'] - mbr['min_lon'])
        gps_seq[:, 1] = (gps_seq[:, 1] - mbr['min_lat']) / (mbr['max_lat'] - mbr['min_lat'])


        time_seq = data['time_seq']
        labels.append(time_seq[-1] - time_seq[0])
        time_seq = [datetime.fromtimestamp(t) for t in data['time_seq']]
        time_fea = torch.as_tensor(data['time_fea'], dtype=torch.float32)
        fwd_dist = torch.as_tensor(data['fwd_dist'] / 1000, dtype=torch.float32)
        bk_dist = torch.as_tensor(data['bk_dist'] / 1000, dtype=torch.float32)
        fwd_az = torch.as_tensor(data['fwd_az'] / 180, dtype=torch.float32)
        bk_az = torch.as_tensor(data['bk_az'] / 180, dtype=torch.float32)

        traj_k = torch.cat([gps_seq, fwd_dist.unsqueeze(-1), bk_dist.unsqueeze(-1), fwd_az.unsqueeze(-1), bk_az.unsqueeze(-1)], axis=1)
        traj_x[k, :traj_len_s5_k] = traj_k
        time_x[k, 0] = time_fea[0]
        time_x[k, 1:traj_len_s5_k] = 0.5  # set the mask time feature as the final time feature

        patch_len_s2_k = data['patch_len_s2']
        traj_len_s2.append(len(patch_len_s2_k))

        patch_len_s3_k = data['patch_len_s3']
        traj_len_s3.append(len(patch_len_s3_k))

        patch_len_s2.extend(patch_len_s2_k)
        patch_len_s3.extend(patch_len_s3_k)

    data = {
        'x': traj_x,
        'time_x': time_x,
        'traj_len_s5': torch.as_tensor(traj_len_s5, dtype=torch.long),
        'patch_len_s3': torch.as_tensor(patch_len_s3, dtype=torch.long),
        'traj_len_s3': torch.as_tensor(traj_len_s3, dtype=torch.long),
        'patch_len_s2': torch.as_tensor(patch_len_s2, dtype=torch.long),
        'traj_len_s2': torch.as_tensor(traj_len_s2, dtype=torch.long),
    }

    return data, torch.as_tensor(labels, dtype=torch.float32).reshape(-1, 1) / 60


def get_dataloader(args, task='pretrain'):
    if task == 'pretrain':
        t1 = time.time()
        date1 = datetime.fromtimestamp(t1)
        print(f'Start  Load data at: {date1}')

        data_preprocess = DataPreprocess(args)

        train_df, eval_df, _ = data_preprocess.run_split()

        train_dataset = TrajDataset(train_df)
        eval_dataset = TrajDataset(eval_df)

        t2 = time.time()
        date2 = datetime.fromtimestamp(t2)
        print(f'Finish Load data at: {date2}\nCost: {t2 - t1} seconds')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True,
                                  num_workers=8, collate_fn=lambda x: collate_fn_pretrain(x, args.center, args.mbr))
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True,
                                 num_workers=8, collate_fn=lambda x: collate_fn_pretrain(x, args.center, args.mbr))
        return train_loader, eval_loader

    elif task.startswith('travel_time'):
        t1 = time.time()
        date1 = datetime.fromtimestamp(t1)
        print(f'Start  Load data at: {date1}')

        data_preprocess = DataPreprocess(args)

        train_df, eval_df, test_df = data_preprocess.run_split()

        train_dataset = TrajDataset(train_df)
        eval_dataset = TrajDataset(eval_df)
        test_dataset = TrajDataset(test_df)

        t2 = time.time()
        date2 = datetime.fromtimestamp(t2)
        print(f'Finish Load data at: {date2}\nCost: {t2 - t1} seconds')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True, prefetch_factor=8,
                                  num_workers=8, collate_fn=lambda x: collate_fn_tte(x, args.center, args.mbr))
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=8, collate_fn=lambda x: collate_fn_tte(x, args.center, args.mbr))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=8, collate_fn=lambda x: collate_fn_tte(x, args.center, args.mbr))
        return train_loader, eval_loader, test_loader

    elif task.startswith('classification'):
        t1 = time.time()
        date1 = datetime.fromtimestamp(t1)
        print(f'Start  Load data at: {date1}')

        data_preprocess = DataPreprocess(args)

        train_df, eval_df, test_df = data_preprocess.run_split()

        train_dataset = TrajDataset(train_df)
        eval_dataset = TrajDataset(eval_df)
        test_dataset = TrajDataset(test_df)

        t2 = time.time()
        date2 = datetime.fromtimestamp(t2)
        print(f'Finish Load data at: {date2}\nCost: {t2 - t1} seconds')

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True,
                                  num_workers=8, collate_fn=lambda x: collate_fn_cls(x, args.center, args.mbr))
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=8, collate_fn=lambda x: collate_fn_cls(x, args.center, args.mbr))
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                 num_workers=8, collate_fn=lambda x: collate_fn_cls(x, args.center, args.mbr))
        return train_loader, eval_loader, test_loader

    elif task.startswith('similarity'):
        raw_database_file_name = os.path.join(data_preprocess.data_save_dir, 'database_traj.parquet')
        database_file_name = os.path.join(data_preprocess.data_save_dir, 'database_traj.pkl')

        raw_label_file_name = os.path.join(data_preprocess.data_save_dir, 'query_sim_traj.parquet')
        label_file_name = os.path.join(data_preprocess.data_save_dir, 'query_sim_traj.pkl')

        raw_query_file_name = os.path.join(data_preprocess.data_save_dir, 'query_traj.parquet')
        query_file_name = os.path.join(data_preprocess.data_save_dir, 'query_traj.pkl')

        database_dataset = data_preprocess.run(database_file_name, raw_database_file_name)
        database_dataset = TrajDataset(database_dataset)

        label_dataset = data_preprocess.run(label_file_name, raw_label_file_name)[:1000]
        label_dataset = TrajDataset(label_dataset)

        query_dataset = data_preprocess.run(query_file_name, raw_query_file_name)[:1000]
        query_dataset = TrajDataset(query_dataset)

        database_loader = DataLoader(database_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                     num_workers=8, collate_fn=lambda x: collate_fn_inference(x, args.center, args.mbr))
        label_loader = DataLoader(label_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                  num_workers=8, collate_fn=lambda x: collate_fn_inference(x, args.center, args.mbr))
        query_loader = DataLoader(query_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                  num_workers=8, collate_fn=lambda x: collate_fn_inference(x, args.center, args.mbr))
        return database_loader, label_loader, query_loader
