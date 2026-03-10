"""
语义距离与结构距离统计分析模块（极简特征分析）
对成功预测的链接样本进行统计，评估模型更依赖语义相似性还是结构相似性
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import ttest_ind

from src.utils.config import PROCESSED_DIR
from src.utils.logger import logger

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_pairwise_distances(
    pairs: np.ndarray,
    sem_features: np.ndarray,
    struct_features: np.ndarray,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算节点对的语义距离与结构距离

    Args:
        pairs: 节点对索引, shape (N, 2)
        sem_features: 语义特征矩阵, shape (N_nodes, D_sem)
        struct_features: 结构特征矩阵, shape (N_nodes, D_struct)
        metric: 距离度量方式 ('cosine' 或 'euclidean')

    Returns:
        (sem_distances, struct_distances), 各 shape (N,)
    """
    dist_fn = cosine if metric == "cosine" else euclidean

    sem_dists = np.array([
        dist_fn(sem_features[u], sem_features[v])
        for u, v in pairs
    ])
    struct_dists = np.array([
        dist_fn(struct_features[u], struct_features[v])
        for u, v in pairs
    ])

    return sem_dists, struct_dists

def analyze_successful_predictions(
    test_pairs: np.ndarray,
    test_labels: np.ndarray,
    pred_probs: np.ndarray,
    sem_features: np.ndarray,
    struct_features: np.ndarray,
    threshold: float = 0.5,
    metric: str = "cosine",
    save_dir: Optional[Path] = None,
) -> Dict:
    """
    对成功预测的正样本链接进行语义距离与结构距离统计分析

    Args:
        test_pairs: 测试节点对, shape (N, 2)
        test_labels: 真实标签, shape (N,)
        pred_probs: 预测概率, shape (N,)
        sem_features: 语义特征矩阵
        struct_features: 结构特征矩阵
        threshold: 预测阈值
        metric: 距离度量
        save_dir: 结果保存路径

    Returns:
        分析结果字典
    """
    save_dir = save_dir or RESULTS_DIR

    preds = (pred_probs >= threshold).astype(int)

    # 成功预测的正样本：真实为1且预测为1
    tp_mask = (test_labels == 1) & (preds == 1)
    # 成功预测的负样本：真实为0且预测为0
    tn_mask = (test_labels == 0) & (preds == 0)
    # 失败样本
    fp_mask = (test_labels == 0) & (preds == 1)
    fn_mask = (test_labels == 1) & (preds == 0)

    logger.info(
        f"预测结果: TP={{tp_mask.sum()}}, TN={{tn_mask.sum()}}, "
        f"FP={{fp_mask.sum()}}, FN={{fn_mask.sum()}}"
    )

    # 计算所有样本的距离
    sem_dists, struct_dists = compute_pairwise_distances(
        test_pairs, sem_features, struct_features, metric
    )

    results = {}

    for name, mask in [("TP", tp_mask), ("TN", tn_mask), ("FP", fp_mask), ("FN", fn_mask)]:
        if mask.sum() == 0:
            continue
        results[name] = {
            "count": int(mask.sum()),
            "mean_sem_dist": float(sem_dists[mask].mean()),
            "std_sem_dist": float(sem_dists[mask].std()),
            "mean_struct_dist": float(struct_dists[mask].mean()),
            "std_struct_dist": float(struct_dists[mask].std()),
        }

    # 核心分析：TP 样本中语义 vs 结构距离
    if "TP" in results:
        tp_sem = sem_dists[tp_mask]
        tp_struct = struct_dists[tp_mask]
        mean_sem = results["TP"]["mean_sem_dist"]
        mean_struct = results["TP"]["mean_struct_dist"]

        logger.info(f"\n{{'='*50}}")
        logger.info(f"成功预测正样本（TP）分析 [距离度量: {{metric}}]")
        logger.info(f"  平均语义距离:  {{mean_sem:.4f}} ± {{results['TP']['std_sem_dist']:.4f}}")
        logger.info(f"  平均结构距离:  {{mean_struct:.4f}} ± {{results['TP']['std_struct_dist']:.4f}}")

        if mean_sem < mean_struct:
            ratio = mean_struct / (mean_sem + 1e-8)
            logger.info(f"  → 语义距离显著低于结构距离（比率={{ratio:.2f}}x）")
            logger.info(f"  → 模型主要捕捉「语义相近」的技术概念之间的内生联结（语义驱动）")
            results["driving_mechanism"] = "semantic"
        else:
            ratio = mean_sem / (mean_struct + 1e-8)
            logger.info(f"  → 结构距离低于语义距离（比率={{ratio:.2f}}x）")
            logger.info(f"  → 模型更加倚重「网络结构位置相近」或「重要性相似」的节点关系（结构驱动）")
            results["driving_mechanism"] = "structural"

        # t 检验
        t_stat, p_value = ttest_ind(tp_sem, tp_struct)
        results["t_test"] = {"t_stat": float(t_stat), "p_value": float(p_value)}
        logger.info(f"  独立样本 t 检验: t={{t_stat:.4f}}, p={{p_value:.4f}}")
        logger.info(f"{{'='*50}}\n")

    # 保存分析结果
    results_df = pd.DataFrame([
        {
            "sample_type": k,
            "count": v["count"],
            "mean_sem_dist": v.get("mean_sem_dist", np.nan),
            "mean_struct_dist": v.get("mean_struct_dist", np.nan),
        }
        for k, v in results.items()
        if isinstance(v, dict) and "count" in v
    ])
    save_path = save_dir / "distance_analysis.csv"
    results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    logger.info(f"距离分析结果已保存: {{save_path}}")

    _plot_distance_comparison(
        sem_dists[tp_mask] if tp_mask.sum() > 0 else np.array([]),
        struct_dists[tp_mask] if tp_mask.sum() > 0 else np.array([]),
        metric=metric,
        save_path=save_dir / "distance_comparison.png",
    )

    return results

def _plot_distance_comparison(
    sem_dists: np.ndarray,
    struct_dists: np.ndarray,
    metric: str = "cosine",
    save_path: Optional[Path] = None,
) -> None:
    """绘制成功预测正样本的语义/结构距离分布对比图"""
    if len(sem_dists) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 直方图对比
    axes[0].hist(sem_dists, bins=30, alpha=0.6, color="#4C72B0", label="Semantic Distance")
    axes[0].hist(struct_dists, bins=30, alpha=0.6, color="#DD8452", label="Structural Distance")
    axes[0].axvline(sem_dists.mean(), color="#4C72B0", linestyle="--",
                    label=f"Sem Mean={{sem_dists.mean():.3f}}")
    axes[0].axvline(struct_dists.mean(), color="#DD8452", linestyle="--",
                    label=f"Struct Mean={{struct_dists.mean():.3f}}")
    axes[0].set_title(f"Distance Distribution (TP Samples, metric={{metric}})")
    axes[0].set_xlabel("Distance")
    axes[0].legend()

    # 箱线图对比
    axes[1].boxplot(
        [sem_dists, struct_dists],
        labels=["Semantic", "Structural"],
        patch_artist=True,
        boxprops=dict(facecolor="#4C72B0", alpha=0.6),
    )
    axes[1].set_title("Distance Boxplot (TP Samples)")
    axes[1].set_ylabel("Distance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.info(f"距离对比图已保存: {{save_path}}")
    plt.close()