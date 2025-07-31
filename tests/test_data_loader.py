# Unit test for the data loading pipeline
# tests/test_data_loader.py
import sys, os, importlib

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.data.data_main import get_data_loaders


def main():
    config_path = "configs.experiments.hpa_mer_mosi_full"
    print(f"--- Testing DataLoader with config: {config_path} ---")
    Config = getattr(importlib.import_module(config_path), 'Config')
    config = Config()

    # 测试开关1：使用Neutral
    config.MAP_NEUTRAL_TO_POSITIVE = True
    print("\n--- Test Case 1: Including Neutral (mapping to Positive) ---")
    train_loader, _, _ = get_data_loaders(config)
    print(f"Total training samples: {len(train_loader.dataset)}")
    batch = next(iter(train_loader))
    print("Batch loaded. Classification labels:", batch['classification_label'])

    # 测试开关2：不使用Neutral
    config.MAP_NEUTRAL_TO_POSITIVE = False
    print("\n--- Test Case 2: Excluding Neutral ---")
    train_loader, _, _ = get_data_loaders(config)
    print(f"Total training samples: {len(train_loader.dataset)}")
    batch = next(iter(train_loader))
    print("Batch loaded. Classification labels:", batch['classification_label'])

    print("\n--- DataLoader Test PASSED ---")


if __name__ == '__main__':
    main()