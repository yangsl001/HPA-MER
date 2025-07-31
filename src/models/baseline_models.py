# src/models/baseline_models.py
import torch.nn as nn
import torch


# 这是一个简化的MulT实现，作为基线
class MulT(nn.Module):
    def __init__(self, config):
        super(MulT, self).__init__()
        # ... (这里可以放置MulT的完整实现，它也应该从config读取参数) ...
        # 为了演示，我们先用一个非常简单的模型代替
        d_model = config.D_MODEL
        self.proj_t = nn.Linear(config.TEXT_EMBED_DIM, d_model)
        self.proj_a = nn.Linear(config.ACOUSTIC_FEAT_DIM, d_model)
        self.proj_v = nn.Linear(config.VISUAL_FEAT_DIM, d_model)
        self.out_reg = nn.Linear(d_model * 3, len(config.REGRESSION_TARGETS)) if config.USE_REGRESSION else None
        self.out_cls = nn.Linear(d_model * 3, config.NUM_CLASSES) if config.USE_CLASSIFICATION else None

    def forward(self, text, audio, vision, **kwargs):
        feat_t = self.proj_t(text).mean(dim=1)
        feat_a = self.proj_a(audio).mean(dim=1)
        feat_v = self.proj_v(vision).mean(dim=1)
        combined = torch.cat([feat_t, feat_a, feat_v], dim=1)

        outputs = {}
        if self.out_reg: outputs['regression'] = self.out_reg(combined)
        if self.out_cls: outputs['classification'] = self.out_cls(combined)
        return outputs
# Implementations of baseline models like MulT
