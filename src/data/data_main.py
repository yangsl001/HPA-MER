# src/data/data_main.py
from torch.utils.data import DataLoader
from .dataset_loader import UnifiedDataset
import torch

def get_data_loaders(config):
    print(f"\n======== Creating DataLoaders for '{config.DATASET_NAME}' ========")

    train_dataset = UnifiedDataset(config, 'train')
    valid_dataset = UnifiedDataset(config, 'valid')
    test_dataset = UnifiedDataset(config, 'test')

    def collate_fn(batch):
        # batch 是一个 list of dicts, e.g., [{'id':..,'text_features':..,'regression_labels':..,'classification_label':..}, ...]
        collated_batch = {}
        # 获取所有键
        keys = batch[0].keys()
        for key in keys:
            if key == 'regression_labels':
                # 将回归标签的字典结构保持下来
                collated_batch[key] = {r_key: torch.stack([d[key][r_key] for d in batch]) for r_key in batch[0][key]}
            elif isinstance(batch[0][key], torch.Tensor):
                collated_batch[key] = torch.stack([d[key] for d in batch])
            else:  # for 'id'
                collated_batch[key] = [d[key] for d in batch]
        return collated_batch

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
                              collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
                             collate_fn=collate_fn)

    print("=" * 60)
    return train_loader, valid_loader, test_loader
# Main entry point for creating data loaders
