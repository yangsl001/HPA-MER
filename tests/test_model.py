# tests/test_model.py (修正打印逻辑)
import sys, os, importlib, torch, numpy as np

# --- 设置路径 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from src.models import get_model


def main():
    config_path = "configs.experiments.hpa_mer_mosi_full"
    print(f"--- Testing Model with config: {config_path} ---")
    Config = getattr(importlib.import_module(config_path), 'Config')
    config = Config()

    try:
        model = get_model(config).to(config.DEVICE)
        print(f"Model '{config.MODEL_NAME}' initialized successfully.")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        bs = 4
        dummy_text = torch.randn(bs, 50, config.TEXT_EMBED_DIM).to(config.DEVICE)
        dummy_audio = torch.randn(bs, 100, config.ACOUSTIC_FEAT_DIM).to(config.DEVICE)
        dummy_vision = torch.randn(bs, 80, config.VISUAL_FEAT_DIM).to(config.DEVICE)

        # --- Test train mode ---
        print("\n--- Train Mode Output ---")
        model.train()
        outputs_train = model(dummy_text, dummy_audio, dummy_vision, use_cma=True)

        # --- 新的、更健壮的打印逻辑 ---
        for key, val in outputs_train.items():
            print(f"  Key '{key}':")
            if isinstance(val, torch.Tensor):
                print(f"    Type: Tensor, Shape: {val.shape}")
            elif isinstance(val, dict):
                print(f"    Type: Dict, Keys: {list(val.keys())}")
            elif isinstance(val, tuple):
                print(f"    Type: Tuple, Content: ({type(val[0])}, {type(val[1])})")
            else:
                print(f"    Type: {type(val)}, Value: {val}")
        # --- 打印逻辑结束 ---

        # 验证输出
        assert isinstance(outputs_train, dict)
        assert 'regression' in outputs_train and 'classification' in outputs_train
        assert 'cma_info' in outputs_train
        assert outputs_train['regression'].shape == (bs, len(config.REGRESSION_TARGETS))
        assert outputs_train['classification'].shape == (bs, config.NUM_CLASSES)
        print("\nTrain mode output structure and shapes are correct.")

        # --- Test eval mode ---
        print("\n--- Eval Mode Output ---")
        model.eval()
        with torch.no_grad():
            outputs_eval = model(dummy_text, dummy_audio, dummy_vision)

        for key, val in outputs_eval.items():
            print(f"  Key '{key}':")
            if isinstance(val, torch.Tensor):
                print(f"    Type: Tensor, Shape: {val.shape}")

        assert isinstance(outputs_eval, dict)
        assert 'regression' in outputs_eval and 'classification' in outputs_eval
        print("\nEval mode output structure and shapes are correct.")

        print("\n--- Model Test PASSED! ---")

    except Exception as e:
        print(f"\n--- Model Test FAILED: {e} ---")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()