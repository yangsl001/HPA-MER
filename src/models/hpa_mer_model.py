# src/models/hpa_mer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# --- 辅助模块 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


# --- HPA-MER 核心子模块 ---
class UEICD_Module(nn.Module):
    def __init__(self, config):
        super(UEICD_Module, self).__init__()
        encoder_layer = lambda dim: nn.TransformerEncoderLayer(d_model=config.D_MODEL, nhead=config.UEICP_ATTN_HEADS,
                                                               dim_feedforward=dim, dropout=config.UEICP_DROPOUT,
                                                               batch_first=True, activation='gelu')
        self.explicit_encoder = nn.TransformerEncoder(encoder_layer(config.UEICP_EXP_FFN_DIM),
                                                      num_layers=config.UEICP_EXP_LAYERS)
        self.implicit_encoder = nn.TransformerEncoder(encoder_layer(config.UEICP_IMP_FFN_DIM),
                                                      num_layers=config.UEICP_IMP_LAYERS)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.D_MODEL))
        self.pos_encoder = PositionalEncoding(config.D_MODEL, config.UEICP_DROPOUT)

    def forward(self, src_seq):
        cls_tokens = self.cls_token.expand(src_seq.size(0), -1, -1)
        src_with_cls = torch.cat([cls_tokens, src_seq], dim=1).transpose(0, 1)
        src_with_cls_pos = self.pos_encoder(src_with_cls).transpose(0, 1)
        f_exp = self.explicit_encoder(src_with_cls_pos)[:, 0, :]
        f_imp = self.implicit_encoder(src_with_cls_pos)[:, 0, :]
        return f_exp, f_imp


class CACE_Module(nn.Module):
    def __init__(self, config):
        super(CACE_Module, self).__init__()
        d_model = config.D_MODEL
        self.explicit_fusion_layer = nn.MultiheadAttention(d_model, config.CACE_EXP_FUSION_HEADS,
                                                           dropout=config.CACE_DROPOUT, batch_first=True)
        self.exp_fusion_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.scale_factor = 1.0 / math.sqrt(d_model)
        self.final_fusion_norm = nn.LayerNorm(d_model)
        self.final_fusion_linear = nn.Linear(d_model, d_model)

    def forward(self, f_exp_dict, f_imp_dict):
        exp_seq = torch.stack(list(f_exp_dict.values()), dim=1)
        imp_seq = torch.stack(list(f_imp_dict.values()), dim=1)
        z_primary, _ = self.explicit_fusion_layer(self.exp_fusion_query.expand(exp_seq.size(0), -1, -1), exp_seq,
                                                  exp_seq)
        z_primary = z_primary.squeeze(1)
        attn_weights = F.softmax(torch.bmm(z_primary.unsqueeze(1), imp_seq.transpose(1, 2)) * self.scale_factor, dim=-1)
        z_rectify = torch.bmm(attn_weights, imp_seq).squeeze(1)
        combined = z_primary + z_rectify + exp_seq.sum(dim=1) + imp_seq.sum(dim=1)
        z_final = self.final_fusion_linear(self.final_fusion_norm(combined))
        return z_final


# --- HPA-MER 主模型 ---
class HPAMerModel(nn.Module):
    def __init__(self, config):
        super(HPAMerModel, self).__init__()
        self.config = config
        d_model = config.D_MODEL

        # MFE
        self.text_projection = nn.Linear(config.TEXT_EMBED_DIM, d_model)
        self.acoustic_encoder = TemporalEncoder(config.ACOUSTIC_FEAT_DIM, config.MFE_A_HIDDEN_DIM, config.MFE_A_LAYERS,
                                                config.MFE_DROPOUT)
        self.acoustic_projection = nn.Linear(config.MFE_A_HIDDEN_DIM * 2, d_model)
        self.visual_encoder = TemporalEncoder(config.VISUAL_FEAT_DIM, config.MFE_V_HIDDEN_DIM, config.MFE_V_LAYERS,
                                              config.MFE_DROPOUT)
        self.visual_projection = nn.Linear(config.MFE_V_HIDDEN_DIM * 2, d_model)

        # UEICD
        self.ueicd_t = UEICD_Module(config)
        self.ueicd_a = UEICD_Module(config)
        self.ueicd_v = UEICD_Module(config)

        # CACE
        self.cace = CACE_Module(config)

        # --- 输出头 (根据config开关创建) ---
        if config.USE_REGRESSION:
            self.regression_heads = nn.ModuleList([self._create_head(d_model, 1) for _ in config.REGRESSION_TARGETS])
        if config.USE_CLASSIFICATION:
            self.classification_head = self._create_head(d_model, config.NUM_CLASSES)
        # 模态特定头只有在需要时才创建
        if config.USE_MODALITY_SPECIFIC_LOSS or config.USE_CMA:
            self.modality_specific_head_t = nn.Linear(d_model, 1)
            self.modality_specific_head_a = nn.Linear(d_model, 1)
            self.modality_specific_head_v = nn.Linear(d_model, 1)

    def _create_head(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.config.CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(self.config.CACE_DROPOUT),
            nn.Linear(self.config.CLASSIFIER_HIDDEN_DIM, output_dim)
        )

    def forward(self, text_features, acoustic_features, visual_features, use_cma=False, alpha=0.2):
        batch_size = text_features.size(0)
        if use_cma and self.training:
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(batch_size).to(self.config.DEVICE)
            acoustic_features = lam * acoustic_features + (1 - lam) * acoustic_features[index, :]
            visual_features = lam * visual_features + (1 - lam) * visual_features[index, :]

        f_t_seq = self.text_projection(text_features)
        f_a_seq = self.acoustic_projection(self.acoustic_encoder(acoustic_features))
        f_v_seq = self.visual_projection(self.visual_encoder(visual_features))

        ft_exp, ft_imp = self.ueicd_t(f_t_seq)
        fa_exp, fa_imp = self.ueicd_a(f_a_seq)
        fv_exp, fv_imp = self.ueicd_v(f_v_seq)

        z_final = self.cace({'t': ft_exp, 'a': fa_exp, 'v': fv_exp}, {'t': ft_imp, 'a': fa_imp, 'v': fv_imp})

        # --- 最终输出 ---
        outputs = {}
        if self.config.USE_REGRESSION:
            outputs['regression'] = torch.cat([head(z_final) for head in self.regression_heads], dim=1)
        if self.config.USE_CLASSIFICATION:
            outputs['classification'] = self.classification_head(z_final)

        if not self.training:
            return outputs

        # --- 训练时返回的额外信息 ---
        if self.config.USE_MODALITY_SPECIFIC_LOSS or self.config.USE_CMA:
            intra_modal_reps = {'text': ft_exp + ft_imp, 'acoustic': fa_exp + fa_imp, 'visual': fv_exp + fv_imp}
            outputs['intermediate_predictions'] = {
                'text': self.modality_specific_head_t(intra_modal_reps['text']).squeeze(-1),
                'acoustic': self.modality_specific_head_a(intra_modal_reps['acoustic']).squeeze(-1),
                'visual': self.modality_specific_head_v(intra_modal_reps['visual']).squeeze(-1)
            }

        outputs['disentanglement_features'] = {'text': (ft_exp, ft_imp), 'acoustic': (fa_exp, fa_imp),
                                               'visual': (fv_exp, fv_imp)}

        if use_cma:
            outputs['cma_info'] = (lam, index)

        return outputs
# Definition of the HPA-MER model architecture
