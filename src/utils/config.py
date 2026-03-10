"""
全局配置参数模块
统一管理所有模块的超参数、路径与开关
"""

from pathlib import Path

# ─────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "experiments" / "results"

# 自动创建目录
for _dir in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 数据预处理配置
# ─────────────────────────────────────────
DATA_CONFIG = {
    "year_start": 2020,
    "year_end": 2024,
    "fields": ["TI", "AB", "DE", "ID", "SC", "PY", "UT"],
    "encoding": "utf-8-sig",
}

# ─────────────────────────────────────────
# 关键词抽取配置
# ─────────────────────────────────────────
KEYWORD_CONFIG = {
    "use_author_kw": True,   # 使用 DE 字段（作者关键词）
    "use_plus_kw": True,     # 使用 ID 字段（扩展关键词）
    "min_freq": 5,           # 最低出现频次
    "min_len": 2,            # 关键词最短字符数
    "max_len": 50,           # 关键词最长字符数
}

# ─────────────────────────────────────────
# 时序划分配置
# ─────────────────────────────────────────
TEMPORAL_CONFIG = {
    "hist_years": (2020, 2023),    # 历史期（训练集）
    "future_years": (2024, 2024),  # 未来期（测试集）
}

# ─────────────────────────────────────────
# 负采样配置
# ─────────────────────────────────────────
SAMPLING_CONFIG = {
    "neg_ratio": 1.0,    # 负样本倍率（1:1）
    "seed": 42,
}

# ─────────────────────────────────────────
# BERT / SciBERT 配置
# ─────────────────────────────────────────
BERT_CONFIG = {
    "model_name": "allenai/scibert_scivocab_uncased",
    "hidden_dim": 768,
    "batch_size": 32,
    "max_length": 64,
}

# ─────────────────────────────────────────
# 特征融合配置
# ─────────────────────────────────────────
FEATURE_CONFIG = {
    "semantic_dim": 768 * 2,    # BERT语义 + 题名语境
    "struct_hidden_dim": 64,
    "struct_out_dim": 32,
}

# ─────────────────────────────────────────
# GraphBERT 模型配置
# ─────────────────────────────────────────
GRAPHBERT_CONFIG = {
    "semantic_dim": 768 * 2,    # 与 FEATURE_CONFIG 对应
    "struct_dim": 32,           # 由 feature_fusion 动态覆盖
    "hidden_dim": 256,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.3,
}

# ─────────────────────────────────────────
# 基线模型配置
# ─────────────────────────────────────────
BASELINE_CONFIG = {
    "hidden_dim": 256,
    "dropout": 0.3,
}

# ─────────────────────────────────────────
# 训练配置
# ─────────────────────────────────────────
TRAIN_CONFIG = {
    "epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "patience": 20,       # 早停轮数
}

# ─────────────────────────────────────────
# SHAP 可解释性配置
# ─────────────────────────────────────────
SHAP_CONFIG = {
    "background_size": 100,
    "n_explain": 200,
}