"""
模型训练与评估模块
支持 GraphBERT 及各基线模型的统一训练/评估流程
指标：AUC-ROC、AUC-PR、Precision@K、Recall@K、F1
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from typing import Dict, List, Tuple, Optional

from src.utils.config import TRAIN_CONFIG
from src.utils.logger import logger


# ─────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    k_list: List[int] = None,
) -> Dict[str, float]:
    """
    计算链接预测评估指标

    Args:
        y_true: 真实标签 (0/1)
        y_score: 预测分数（概率）
        threshold: 二值化阈值
        k_list: Precision@K 的 K 值列表

    Returns:
        指标字典
    """
    k_list = k_list or [10, 50, 100]
    metrics = {}

    # AUC-ROC
    metrics["auc_roc"] = roc_auc_score(y_true, y_score)

    # AUC-PR（平均精度）
    metrics["auc_pr"] = average_precision_score(y_true, y_score)

    # F1（基于阈值）
    y_pred = (y_score >= threshold).astype(int)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # Precision@K
    sorted_idx = np.argsort(-y_score)
    for k in k_list:
        top_k_idx = sorted_idx[:k]
        metrics[f"precision@{k}"] = y_true[top_k_idx].mean()

    # Recall@K
    total_pos = y_true.sum()
    for k in k_list:
        top_k_idx = sorted_idx[:k]
        metrics[f"recall@{k}"] = y_true[top_k_idx].sum() / max(total_pos, 1)

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """格式化指标输出"""
    parts = []
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)


# ─────────────────────────────────────────────
# GraphBERT 训练器
# ─────────────────────────────────────────────

class GraphBertTrainer:
    """
    GraphBERT 链接预测模型训练器

    支持：
    - 混合精度训练（FP16）
    - 学习率调度（WarmupCosine）
    - 早停机制
    - 检查点保存/加载
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        lr: float = None,
        weight_decay: float = None,
        epochs: int = None,
        patience: int = None,
        save_dir: str = "experiments/results",
    ):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr or TRAIN_CONFIG["lr"]
        self.weight_decay = weight_decay or TRAIN_CONFIG["weight_decay"]
        self.epochs = epochs or TRAIN_CONFIG["epochs"]
        self.patience = patience or TRAIN_CONFIG["patience"]
        self.save_dir = save_dir

        self.model = self.model.to(self.device)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_auc = 0.0
        self.patience_counter = 0

        import os
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"训练器初始化完成，设备: {self.device}，学习率: {self.lr}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch，返回平均损失"""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            u_feats, u_pos, v_feats, v_pos, labels = [b.to(self.device) for b in batch]

            self.optimizer.zero_grad()
            logits = self.model(u_feats, u_pos, v_feats, v_pos)
            loss = self.criterion(logits, labels.float())
            loss.backward()

            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """评估模型，返回（损失, 指标字典）"""
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        for batch in dataloader:
            u_feats, u_pos, v_feats, v_pos, labels = [b.to(self.device) for b in batch]
            logits = self.model(u_feats, u_pos, v_feats, v_pos)
            loss = self.criterion(logits, labels.float())
            total_loss += loss.item()

            all_logits.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        y_score = np.concatenate(all_logits)
        y_true = np.concatenate(all_labels)
        metrics = compute_metrics(y_true, y_score)
        avg_loss = total_loss / len(dataloader)

        return avg_loss, metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler=None,
    ) -> Dict[str, List]:
        """
        完整训练流程

        Returns:
            训练历史字典 {train_loss, val_loss, val_auc_roc, ...}
        """
        history = {"train_loss": [], "val_loss": [], "val_auc_roc": [], "val_auc_pr": []}

        logger.info(f"开始训练，共 {self.epochs} 个 epoch...")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.evaluate(val_loader)

            if scheduler:
                scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc_roc"].append(val_metrics["auc_roc"])
            history["val_auc_pr"].append(val_metrics["auc_pr"])

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{self.epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"{format_metrics(val_metrics)}  [{elapsed:.1f}s]"
            )

            # 早停与保存最优模型
            if val_metrics["auc_roc"] > self.best_val_auc:
                self.best_val_auc = val_metrics["auc_roc"]
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                logger.info(f"  >> 最优模型已保存（val_auc={self.best_val_auc:.4f}）")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"早停触发（连续 {self.patience} 个 epoch 无改善）")
                    break

        logger.info(f"训练完成！最优验证 AUC-ROC: {self.best_val_auc:.4f}")
        return history

    def save_checkpoint(self, filename: str) -> None:
        """保存模型检查点"""
        import os
        path = os.path.join(self.save_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_auc": self.best_val_auc,
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        """加载模型检查点"""
        import os
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_auc = checkpoint.get("best_val_auc", 0.0)
        logger.info(f"模型已从 {path} 加载（best_val_auc={self.best_val_auc:.4f}）")


# ─────────────────────────────────────────────
# 基线模型评估（统一接口）
# ─────────────────────────────────────────────

def evaluate_heuristic(
    scores: np.ndarray,
    labels: np.ndarray,
    method_name: str,
) -> Dict[str, float]:
    """评估启发式基线，返回指标字典"""
    metrics = compute_metrics(labels, scores)
    logger.info(f"[{method_name}] {format_metrics(metrics)}")
    return metrics


def compare_all_methods(
    results: Dict[str, Dict[str, float]],
) -> None:
    """打印所有方法的对比表格"""
    import pandas as pd
    df = pd.DataFrame(results).T
    df = df.sort_values("auc_roc", ascending=False)
    logger.info(f"\n{'='*60}\n模型对比结果:\n{df.to_string()}\n{'='*60}")


if __name__ == "__main__":
    logger.info("训练模块加载成功，请通过 scripts/run_pipeline.sh 启动完整训练流程")
