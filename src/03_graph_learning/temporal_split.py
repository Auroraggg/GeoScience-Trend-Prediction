"""
时序划分与链接预测样本构造模块
严格按照"历史训练、未来验证"策略构造正负样本，防止时间信息泄露
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from typing import Tuple, List, Dict

from src.utils.config import PROCESSED_DIR, PREPROCESS_CONFIG, TRAIN_CONFIG
from src.utils.logger import logger

def get_future_edges(
    G_hist: nx.Graph,
    G_full: nx.Graph,
) -> List[Tuple[str, str]]:
    """
    提取未来新增边（2024年新出现的共现关系）
    只保留两端节点均在历史图中存在的边（transductive setting）

    Args:
        G_hist: 历史期图（训练集）
        G_full: 全量图（含未来边）

    Returns:
        未来正样本边列表
    """
    hist_nodes = set(G_hist.nodes())
    hist_edges = set(G_hist.edges())
    # 确保对称
    hist_edges_sym = hist_edges | {(v, u) for u, v in hist_edges}

    future_edges = []
    for u, v in G_full.edges():
        if u == v:
            continue
        if (u, v) in hist_edges_sym or (v, u) in hist_edges_sym:
            continue
        # 两端节点必须在历史图中存在
        if u in hist_nodes and v in hist_nodes:
            future_edges.append((u, v))

    logger.info(f"未来正样本边数（两端节点均在历史图中）: {len(future_edges)}")
    return future_edges

def build_train_samples(
    G_hist: nx.Graph,
    valid_kws: List[str],
    neg_pos_ratio: int = None,
    val_ratio: float = None,
    seed: int = None,
) -> Dict:
    """
    构造训练集正负样本（基于历史图）

    Args:
        G_hist: 历史期图
        valid_kws: 有效关键词列表（有特征的节点）
        neg_pos_ratio: 负正样本比例
        val_ratio: 从训练集中划分出验证集的比例
        seed: 随机种子

    Returns:
        dict with keys: train_pos, train_neg, val_pos, val_neg
    """
    from src.03_graph_learning.negative_sampling import sample_negatives

    neg_pos_ratio = neg_pos_ratio or TRAIN_CONFIG["neg_pos_ratio"]
    val_ratio = val_ratio or TRAIN_CONFIG["val_ratio"]
    seed = seed or TRAIN_CONFIG["seed"]
    np.random.seed(seed)

    # 历史正样本：历史图中存在的边，两端节点均有特征
    valid_set = set(valid_kws)
    pos_edges = [
        (u, v) for u, v in G_hist.edges()
        if u in valid_set and v in valid_set
    ]
    logger.info(f"历史正样本边数: {len(pos_edges)}")

    # 负采样
    neg_edges = sample_negatives(
        G_hist, valid_kws, n_samples=len(pos_edges) * neg_pos_ratio
    )

    # 打乱并划分训练/验证
    pos_arr = np.array(pos_edges)
    neg_arr = np.array(neg_edges)

    pos_idx = np.random.permutation(len(pos_arr))
    neg_idx = np.random.permutation(len(neg_arr))

    val_pos_n = int(len(pos_arr) * val_ratio)
    val_neg_n = int(len(neg_arr) * val_ratio)

    result = {
        "train_pos": pos_arr[pos_idx[val_pos_n:]].tolist(),
        "train_neg": neg_arr[neg_idx[val_neg_n:]].tolist(),
        "val_pos": pos_arr[pos_idx[:val_pos_n]].tolist(),
        "val_neg": neg_arr[neg_idx[:val_neg_n]].tolist(),
    }

    logger.info(
        f"训练集: {len(result['train_pos'])} 正 / {len(result['train_neg'])} 负"
    )
    logger.info(
        f"验证集: {len(result['val_pos'])} 正 / {len(result['val_neg'])} 负"
    )
    return result

def build_test_samples(
    future_edges: List[Tuple[str, str]],
    G_hist: nx.Graph,
    valid_kws: List[str],
    neg_pos_ratio: int = None,
    seed: int = None,
) -> Dict:
    """
    构造测试集样本（基于未来新增边）

    Args:
        future_edges: 未来正样本边
        G_hist: 历史图（用于确保负样本不在历史图中）
        valid_kws: 有效关键词列表
        neg_pos_ratio: 负正样本比例
        seed: 随机种子

    Returns:
        dict with keys: test_pos, test_neg
    """
    from src.03_graph_learning.negative_sampling import sample_negatives

    neg_pos_ratio = neg_pos_ratio or TRAIN_CONFIG["neg_pos_ratio"]
    seed = seed or TRAIN_CONFIG["seed"]
    np.random.seed(seed)

    valid_set = set(valid_kws)
    test_pos = [(u, v) for u, v in future_edges if u in valid_set and v in valid_set]

    test_neg = sample_negatives(
        G_hist, valid_kws, n_samples=len(test_pos) * neg_pos_ratio, seed=seed + 1
    )

    logger.info(f"测试集: {len(test_pos)} 正 / {len(test_neg)} 负")
    return {"test_pos": test_pos, "test_neg": test_neg}

def save_samples(samples: Dict, filename: str = "link_prediction_samples.pkl") -> Path:
    """保存样本集"""
    path = PROCESSED_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    logger.info(f"样本已保存: {path}")
    return path

def load_samples(filename: str = "link_prediction_samples.pkl") -> Dict:
    """加载样本集"""
    path = PROCESSED_DIR / filename
    with open(path, "rb") as f:
        samples = pickle.load(f)
    logger.info(f"样本已加载，keys: {list(samples.keys())}")
    return samples
