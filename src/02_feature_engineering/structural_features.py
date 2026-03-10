"""
结构特征提取模块
计算关键词共现网络中每个节点的四类结构特征：
度(Degree)、PageRank、聚类系数(Clustering)、中介中心性(Betweenness)
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, List

from src.utils.config import PROCESSED_DIR
from src.utils.logger import logger

def compute_degree(G: nx.Graph, normalize: bool = True) -> Dict[str, float]:
    """
    计算节点加权度

    Args:
        G: 共现图
        normalize: 是否归一化到 [0, 1]

    Returns:
        节点->度 字典
    """
    degree = dict(G.degree(weight="weight"))
    if normalize and len(degree) > 1:
        max_val = max(degree.values())
        min_val = min(degree.values())
        denom = max_val - min_val if max_val != min_val else 1.0
        degree = {k: (v - min_val) / denom for k, v in degree.items()}
    return degree

def compute_pagerank(G: nx.Graph, alpha: float = 0.85) -> Dict[str, float]:
    """
    计算节点 PageRank 值

    Args:
        G: 共现图
        alpha: 阻尼系数

    Returns:
        节点->PageRank 字典
    """
    return nx.pagerank(G, alpha=alpha, weight="weight")

def compute_clustering(G: nx.Graph) -> Dict[str, float]:
    """
    计算节点加权聚类系数

    Args:
        G: 共现图

    Returns:
        节点->聚类系数 字典
    """
    return nx.clustering(G, weight="weight")

def compute_betweenness(G: nx.Graph, normalize: bool = True, k: int = None) -> Dict[str, float]:
    """
    计算节点中介中心性
    ��于大型图使用近似算法（k采样）

    Args:
        G: 共现图
        normalize: 是否归一化
        k: 近似计算时采样节点数，None 表示精确计算

    Returns:
        节点->中介中心性 字典
    """
    n = G.number_of_nodes()
    # 节点数 > 2000 时自动切换为近似计算
    if k is None and n > 2000:
        k = min(500, n)
        logger.info(f"图规模较大({n}节点)，使用近似中介中心性(k={k})")

    return nx.betweenness_centrality(G, normalized=normalize, weight="weight", k=k)

def build_structural_feature_matrix(G: nx.Graph) -> pd.DataFrame:
    """
    构建完整结构特征矩阵

    Args:
        G: 历史期共现图

    Returns:
        DataFrame，index 为关键词，列为 [degree, pagerank, clustering, betweenness]
    """
    logger.info("开始提取结构特征...")

    nodes = list(G.nodes())
    logger.info(f"共 {len(nodes)} 个节点")

    degree = compute_degree(G)
    logger.info("  ✓ Degree 计算完成")

    pagerank = compute_pagerank(G)
    logger.info("  ✓ PageRank 计算完成")

    clustering = compute_clustering(G)
    logger.info("  ✓ Clustering 计算完成")

    betweenness = compute_betweenness(G)
    logger.info("  ✓ Betweenness 计算完成")

    struct_df = pd.DataFrame({
        "keyword": nodes,
        "degree": [degree.get(n, 0.0) for n in nodes],
        "pagerank": [pagerank.get(n, 0.0) for n in nodes],
        "clustering": [clustering.get(n, 0.0) for n in nodes],
        "betweenness": [betweenness.get(n, 0.0) for n in nodes],
    }).set_index("keyword")

    logger.info(f"结构特征矩阵构建完成，形状: {struct_df.shape}")
    return struct_df

def save_structural_features(struct_df: pd.DataFrame, filename: str = "structural_features.csv") -> None:
    """保存结构特征到 CSV"""
    out_path = PROCESSED_DIR / filename
    struct_df.to_csv(out_path, encoding="utf-8-sig")
    logger.info(f"结构特征已保存: {out_path}")

def load_structural_features(filename: str = "structural_features.csv") -> pd.DataFrame:
    """从 CSV 加载结构特征"""
    path = PROCESSED_DIR / filename
    df = pd.read_csv(path, index_col="keyword", encoding="utf-8-sig")
    logger.info(f"结构特征已加载: {path}, 形状: {df.shape}")
    return df

if __name__ == "__main__":
    from src.01_preprocessing.network_builder import load_graphs

    G_hist, _ = load_graphs()
    struct_df = build_structural_feature_matrix(G_hist)
    save_structural_features(struct_df)
    print(struct_df.describe())