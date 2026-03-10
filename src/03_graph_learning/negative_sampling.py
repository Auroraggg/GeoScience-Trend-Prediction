"""
负样本采样模块
高效从未连接节点对中抽取负样本，避免时间信息泄露
"""

import random
import numpy as np
from typing import List, Set, Tuple

from src.utils.logger import logger

def sample_negatives(
    pos_edges: List[Tuple[str, str]],
    all_nodes: List[str],
    num_neg: int,
    existing_edges: Set[Tuple[str, str]],
    seed: int = 42,
    max_attempts: int = 50,
) -> List[Tuple[str, str]]:
    """
    随机负采样：从所有节点对中抽取不在 existing_edges 中的负样本

    Args:
        pos_edges:      正样本边列表
        all_nodes:      所有可用节点列表
        num_neg:        目标负样本数量
        existing_edges: 需要排除的已有边集合（正样本+历史边）
        seed:           随机种子
        max_attempts:   最大重试倍数（防止稀疏图下死循环）

    Returns:
        neg_edges: 负样本边列表
    """
    random.seed(seed)
    np.random.seed(seed)

    n = len(all_nodes)
    node_array = np.array(all_nodes)
    neg_edges = []
    attempts = 0
    max_total = num_neg * max_attempts

    while len(neg_edges) < num_neg and attempts < max_total:
        # 批量随机采样提升效率
        batch_size = min(num_neg * 2, 10000)
        idx1 = np.random.randint(0, n, batch_size)
        idx2 = np.random.randint(0, n, batch_size)

        for i, j in zip(idx1, idx2):
            if len(neg_edges) >= num_neg:
                break
            u, v = node_array[i], node_array[j]
            if u == v:
                continue
            edge = (min(u, v), max(u, v))
            if edge not in existing_edges:
                neg_edges.append((u, v))
                existing_edges.add(edge)  # 避免重复
        attempts += batch_size

    if len(neg_edges) < num_neg:
        logger.warning(
            f"负采样未达目标：期望 {num_neg}，实际 {len(neg_edges)}。"
            f"图可能过于稠密，建议降低 neg_ratio。"
        )

    return neg_edges

def structured_negative_sampling(
    pos_edges: List[Tuple[str, str]],
    all_nodes: List[str],
    existing_edges: Set[Tuple[str, str]],
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """
    结构化负采样：对每条正样本边，随机替换一个端点
    生成与正样本数量相同的负样本，保持局部拓扑特性

    Args:
        pos_edges:      正样本边列表
        all_nodes:      所有节点列表
        existing_edges: 需要排除的已有边集合
        seed:           随机种子

    Returns:
        neg_edges: 负样本边列表（等量于正样本）
    """
    random.seed(seed)
    np.random.seed(seed)

    neg_edges = []
    node_array = np.array(all_nodes)
    n = len(all_nodes)

    for u, v in pos_edges:
        found = False
        for _ in range(100):
            # 随机替换一端
            if random.random() < 0.5:
                new_u = node_array[np.random.randint(0, n)]
                candidate = (min(new_u, v), max(new_u, v))
                if new_u != v and candidate not in existing_edges:
                    neg_edges.append((new_u, v))
                    existing_edges.add(candidate)
                    found = True
                    break
            else:
                new_v = node_array[np.random.randint(0, n)]
                candidate = (min(u, new_v), max(u, new_v))
                if u != new_v and candidate not in existing_edges:
                    neg_edges.append((u, new_v))
                    existing_edges.add(candidate)
                    found = True
                    break
        if not found:
            # 兜底：随机采一条
            for _ in range(100):
                ru = node_array[np.random.randint(0, n)]
                rv = node_array[np.random.randint(0, n)]
                edge = (min(ru, rv), max(ru, rv))
                if ru != rv and edge not in existing_edges:
                    neg_edges.append((ru, rv))
                    existing_edges.add(edge)
                    break

    return neg_edges
