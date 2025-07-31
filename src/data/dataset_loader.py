# Unified dataset loader class
# src/data/dataset_loader.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import pickle


class UnifiedDataset(Dataset):
    def __init__(self, config, split_type='train'):
        self.config = config
        self.split_type = split_type

        data_path = config.DATA_DIR
        label_df_full = pd.read_csv(os.path.join(data_path, 'label.csv'))

        # 1. 根据分类任务的需求，预先筛选数据
        if config.USE_CLASSIFICATION and not config.MAP_NEUTRAL_TO_POSITIVE:
            # 如果不使用Neutral，则在所有数据中都过滤掉它
            label_df_full = label_df_full[label_df_full[config.CLASSIFICATION_TARGET] != 'Neutral']

        self.metadata = label_df_full[label_df_full['mode'] == split_type].copy()
        self.metadata['id'] = self.metadata['video_id'].astype(str) + '$_$' + self.metadata['clip_id'].astype(str)
        self.metadata.set_index('id', inplace=True)

        with open(os.path.join(data_path, config.DATA_TYPE), 'rb') as f:
            self.features = pickle.load(f)[split_type]

        self.feature_id_to_idx = {id_str: i for i, id_str in enumerate(self.features['id'])}

        valid_ids = set(self.feature_id_to_idx.keys()).intersection(set(self.metadata.index))
        self.sample_ids = sorted(list(valid_ids))

        if config.USE_CLASSIFICATION:
            self.label_to_id = {label: i for i, label in enumerate(config.CLASS_LABELS)}

        print(f"Loaded {len(self.sample_ids)} samples for '{config.DATASET_NAME} - {split_type}'.")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        label_info = self.metadata.loc[sample_id]
        feature_idx = self.feature_id_to_idx[sample_id]

        item = {
            'id': sample_id,
            'text_features': torch.tensor(self.features['text'][feature_idx], dtype=torch.float32),
            'acoustic_features': torch.tensor(self.features['audio'][feature_idx], dtype=torch.float32),
            'visual_features': torch.tensor(self.features['vision'][feature_idx], dtype=torch.float32)
        }

        if self.config.USE_REGRESSION:
            reg_labels = {}
            for key in self.config.REGRESSION_TARGETS:
                value = label_info.get(key)
                if pd.isna(value): value = label_info['label']
                reg_labels[key] = torch.tensor(value, dtype=torch.float32)
            item['regression_labels'] = reg_labels

        if self.config.USE_CLASSIFICATION:
            label_str = label_info[self.config.CLASSIFICATION_TARGET]
            if self.config.MAP_NEUTRAL_TO_POSITIVE and label_str == 'Neutral':
                label_str = 'Positive'  # 将Neutral映射为Positive

            # 转换为二分类ID
            label_id = 1 if label_str == 'Positive' else 0
            item['classification_label'] = torch.tensor(label_id, dtype=torch.long)

        return item