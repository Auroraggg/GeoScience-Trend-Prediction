"""
结构特征提取模块
计算关键词共现网络中各节点的四类核心结构指标：
度(Degree)、PageRank、聚类系数(Clustering)、中介中心性(Betweenness)
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

from src.utils.config import PROCESSED_DIR, STRUCT_CONFIG
from src.utils.logger import logger

def compute_structural_features(
    G: nx.Graph,
    features: List[str] = None,
    normalize: bool = None,
) -> pd.DataFrame:
    """
    计算图中所有节点的结构特征

    Args:
        G: NetworkX 带权无向图
        features: 要计算的特征列表，默认使用 config 配置
        normalize: 是否 Z-score 标准化

    Returns:
        节点结构特征 DataFrame，index 为节点名称
    """
    features = features or STRUCT_CONFIG["features"]
    normalize = normalize if normalize is not None else STRUCT_CONFIG["normalize"]
    nodes = list(G.nodes())

    logger.info(f"开始计算结构特征: {features}，节点数: {len(nodes)}")
    feat_dict = {node: {} for node in nodes}

    if "degree" in features:
        logger.info("  计算 Degree...")
        degree = dict(G.degree(weight="weight"))
        for node in nodes:
            feat_dict[node]["degree"] = degree.get(node, 0)

    if "pagerank" in features:
        logger.info("  计算 PageRank...")
        alpha = STRUCT_CONFIG.get("pagerank_alpha", 0.85)
        pagerank = nx.pagerank(G, alpha=alpha, weight="weight", max_iter=200)
        for node in nodes:
            feat_dict[node]["pagerank"] = pagerank.get(node, 0.0)

    if "clustering" in features:
        logger.info("  计算 Clustering Coefficient...")
        clustering = nx.clustering(G, weight="weight")
        for node in nodes:
            feat_dict[node]["clustering"] = clustering.get(node, 0.0)

    if "betweenness" in features:
        logger.info("  计算 Betweenness Centrality（可能较慢）...")
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        for node in nodes:
            feat_dict[node]["betweenness"] = betweenness.get(node, 0.0)

    struct_df = pd.DataFrame.from_dict(feat_dict, orient="index")
    struct_df.index.name = "keyword"
    struct_df = struct_df[features]  # 保证列顺序

    if normalize:
        logger.info("  标准化结构特征 (Z-score)...")
        scaler = StandardScaler()
        struct_df[features] = scaler.fit_transform(struct_df[features])

    logger.info(f"结构特征计算完成，形状: {struct_df.shape}")
    return struct_df

def get_struct_feature_matrix(
    struct_df: pd.DataFrame,
    node_list: List[str],
) -> np.ndarray:
    """
    根据节点顺序提取结构特征矩阵

    Args:
        struct_df: 结构特征 DataFrame
        node_list: 目标节点列表（定义行顺序）

    Returns:
        np.ndarray of shape (len(node_list), num_features)
    """
    # 对于不在图中的节点，用0填充
    valid_nodes = [n for n in node_list if n in struct_df.index]
    missing = set(node_list) - set(valid_nodes)
    if missing:
        logger.warning(f"{len(missing)} 个节点不在结构特征表中，用0填充")

    mat = []
    for node in node_list:
        if node in struct_df.index:
            mat.append(struct_df.loc[node].values.astype(np.float32))
        else:
            mat.append(np.zeros(struct_df.shape[1], dtype=np.float32))
    return np.vstack(mat)

def save_structural_features(struct_df: pd.DataFrame, filename: str = "structural_features.csv") -> None:
    """保存结构特征到 processed 目录"""
    save_path = PROCESSED_DIR / filename
    struct_df.to_csv(save_path, encoding="utf-8-sig")
    logger.info(f"结构特征已保存: {save_path}")

def load_structural_features(filename: str = "structural_features.csv") -> pd.DataFrame:
    """从 processed 目录加载结构特征"""
    load_path = PROCESSED_DIR / filename
    struct_df = pd.read_csv(load_path, index_col="keyword")
    logger.info(f"结构特征已加载: {struct_df.shape}")
    return struct_df

if __name__ == "__main__":
    from src.01_preprocessing.network_builder import load_graphs

    G_hist, _ = load_graphs()
    struct_df = compute_structural_features(G_hist)
    save_structural_features(struct_df)

    logger.info(f"特征统计:\n{struct_df.describe().to_string()}")