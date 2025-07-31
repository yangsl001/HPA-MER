# run.py
import importlib, sys, os, torch, numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.data.data_main import get_data_loaders
from src.trainers.hpa_mer_trainer import HPA_MER_Trainer


def main():
    # --- 运行实验的唯一开关 ---
    config_path = "configs.experiments.hpa_mer_mosi_full"
    # -----------------------------

    print(f"--- 正在加载配置: {config_path} ---")
    Config = getattr(importlib.import_module(config_path), 'Config')
    config = Config()

    # 设置随机种子
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 加载数据
    train_loader, valid_loader, test_loader = get_data_loaders(config)

    # 创建并启动训练器
    trainer = HPA_MER_Trainer(config)
    trainer.do_train(train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    main()