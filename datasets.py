import os
import numpy as np
from torch.utils.data import Dataset
import torch
from typing import List
import gc
from utils import get_contents_in_dir

class TrafficScopeDataset:
    def __init__(self, data_dir, agg_scales: List[int], indices=None):
        # 加载数据并生成标签
        temporal_data_files = get_contents_in_dir(data_dir, ['.'], ['temporal.npy'])
        temporal_mask_files = get_contents_in_dir(data_dir, ['.'], ['mask.npy'])
        contextual_data_files = get_contents_in_dir(data_dir, ['.'], ['contextual.npy'])
        label_files = get_contents_in_dir(data_dir, ['.'], ['labels.npy'])

        tmp_temporal_data_list = []
        tmp_temporal_mask_data_list = []
        tmp_contextual_data_list = []
        tmp_labels_list = []
        data_len = 0
        
        for idx in range(len(temporal_data_files)):
            temporal_data_path = temporal_data_files[idx]
            temporal_mask_path = temporal_mask_files[idx]
            contextual_data_path = contextual_data_files[idx]
            label_path = label_files[idx]
            
            tmp_temporal_data = np.load(temporal_data_path)
            # tmp_temporal_data = tmp_temporal_data.astype(np.float32)
            tmp_temporal_mask_data = np.load(temporal_mask_path)
            tmp_contextual_data = np.load(contextual_data_path)#, mmap_mode='r')
            # tmp_contextual_data = tmp_contextual_data.astype(np.float32)
            tmp_labels = np.load(label_path)
            
            N = tmp_contextual_data.shape[0]
            # N = 20000
            # dices = np.random.choice(N, size=30000, replace=False)
            ###################################################################
            dices = np.arange(N)
            tmp_temporal_data = tmp_temporal_data[dices]
            tmp_temporal_mask_data = tmp_temporal_mask_data[dices]
            tmp_contextual_data = tmp_contextual_data[dices]
            tmp_labels = tmp_labels[dices]

            tmp_temporal_data_list.append(tmp_temporal_data)
            tmp_temporal_mask_data_list.append(tmp_temporal_mask_data)
            tmp_contextual_data_list.append(tmp_contextual_data)
            tmp_labels_list.append(tmp_labels)
            
            data_len += tmp_temporal_data.shape[0]
            print(f'Loaded {temporal_data_files[idx]}, len: {tmp_temporal_data.shape[0]}')

        # 合并所有数据
        self.temporal_data = np.vstack(tmp_temporal_data_list)
        self.temporal_mask_data = np.vstack(tmp_temporal_mask_data_list)
        self.contextual_data = np.vstack(tmp_contextual_data_list)
        self.labels = np.concatenate(tmp_labels_list)
        
        print(f'Total samples: {data_len}')

        # 生成有效长度
        self.temporal_valid_len = self.temporal_mask_data.shape[1] - \
                                  (self.temporal_mask_data.sum(axis=2) == self.temporal_mask_data.shape[2]).sum(axis=1)

        # 解包上下文特征 N x agg_scale_num x freqs x t --> N x freqs x (agg_scale_num x t)
        concatenate_data = []
        
        for agg_scale in agg_scales:
            
            concatenate_data.append(self.contextual_data[:, agg_scale, :, :])
        self.contextual_data_unpack = np.concatenate(concatenate_data, axis=2)
        
        # 让最后一个维度是特征 N x freqs x (agg_scale_num x t) --> N x (agg_scale_num x t) x freqs
        self.contextual_data_unpack = self.contextual_data_unpack.transpose((0, 2, 1))

        # 生成上下文段 N x (agg_scale_num x t)
        segment_len = self.contextual_data.shape[3]
        self.contextual_segments = np.zeros((self.contextual_data.shape[0],
                                              len(agg_scales)*segment_len))
        for agg_scale_idx, _ in enumerate(agg_scales):
            self.contextual_segments[:, segment_len*agg_scale_idx:segment_len*(agg_scale_idx+1)] = agg_scale_idx

        if indices is not None:
            self.temporal_data = self.temporal_data[indices]
            self.temporal_mask_data = self.temporal_mask_data[indices]
            self.contextual_data_unpack = self.contextual_data_unpack[indices]
            self.contextual_segments = self.contextual_segments[indices]
            self.labels = self.labels[indices]
            print(f'Total samples after indexing: {self.temporal_data.shape[0]}')
            
        print('Dataset loaded successfully')

    def __getitem__(self, idx):
        return (torch.tensor(self.temporal_data[idx], dtype=torch.float),
                torch.tensor(self.temporal_valid_len[idx], dtype=torch.float),
                torch.tensor(self.contextual_data_unpack[idx], dtype=torch.float),
                torch.tensor(self.contextual_segments[idx], dtype=torch.long),
                torch.tensor(int(self.labels[idx]), dtype=torch.long))

    def __len__(self):
        return self.temporal_data.shape[0]
    
if __name__ == '__main__':
    dataset = TrafficScopeDataset('./', [0, 1, 2], [0, 1, 2, 3, 4, 5])
    print(f"Dataset length: {len(dataset)}")
    
    ##############################################################################
    # 测试获取一个样本
    sample = dataset[0]
    print(f"Sample contains {len(sample)} elements:")
    for i, elem in enumerate(sample):
        print(f"  {i}: {type(elem)}, shape: {elem.shape if hasattr(elem, 'shape') else 'scalar'}")
        print(elem)
