"""
BERT / SciBERT 语义嵌入模块
对关键词文本及其题名语境进行批量语义编码，生成高维语义向量
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModel

from src.utils.config import BERT_CONFIG, PROCESSED_DIR
from src.utils.logger import logger


class BERTEmbedder:
    """
    BERT/SciBERT 批量语义嵌入器

    支持两种嵌入策略：
    1. 关键词文本嵌入（直接对关键词字符串编码）
    2. 题名语境嵌入（对包含该关键词的所有题名取平均）
    """

    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None, max_length: int = None):
        self.model_name = model_name or BERT_CONFIG["model_name"]
        self.batch_size = batch_size or BERT_CONFIG["batch_size"]
        self.max_length = max_length or BERT_CONFIG["max_length"]

        # 设备选择
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"加载 BERT 模型: {self.model_name}，设备: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("BERT 模型加载完成")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本列表，返回 [CLS] token 表示

        Args:
            texts: 文本列表

        Returns:
            numpy array，形状 (N, hidden_size)
        """
        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_idx = i // self.batch_size + 1
            if batch_idx % 10 == 0 or batch_idx == total_batches:
                logger.info(f"  编码进度: {batch_idx}/{total_batches} batches")

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)

            # 取 [CLS] token 的最后一层表示
            cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

        return np.vstack(all_embeddings)

    def encode_keywords(self, keywords: List[str]) -> Dict[str, np.ndarray]:
        """
        对关键词文本进行直接编码

        Args:
            keywords: 关键词列表

        Returns:
            {keyword: embedding_vector} 字典
        """
        logger.info(f"编码关键词文本，共 {len(keywords)} 个关键词...")
        embeddings = self.encode(keywords)
        kw2embed = {kw: embeddings[i] for i, kw in enumerate(keywords)}
        logger.info("关键词文本编码完成")
        return kw2embed

    def encode_title_context(self, keywords: List[str], df: pd.DataFrame, title_col: str = "TI") -> Dict[str, np.ndarray]:
        """
        对每个关键词，聚合其出现的所有题名语境，取平均嵌入

        Args:
            keywords: 关键词列表
            df: 含 'keywords' 和题名列的 DataFrame
            title_col: 题名列名（默认 'TI'）

        Returns:
            {keyword: avg_title_embedding} 字典
        """
        logger.info(f"构建关键词→题名倒排索引...")

        # 建立倒排索引
        kw_to_titles: Dict[str, List[str]] = {kw: [] for kw in keywords}
        for _, row in df.iterrows():
            title = str(row.get(title_col, "")).strip()
            if not title:
                continue
            for kw in row.get("keywords", []):
                if kw in kw_to_titles:
                    kw_to_titles[kw].append(title)

        logger.info("计算题名语境平均嵌入...")
        kw2title_embed = {}
        # 收集所有需要编码的唯一题名
        all_titles = list({t for titles in kw_to_titles.values() for t in titles})
        logger.info(f"  共 {len(all_titles)} 个唯一题名需要编码")
        title2embed = {}
        if all_titles:
            title_embeds = self.encode(all_titles)
            title2embed = {t: title_embeds[i] for i, t in enumerate(all_titles)}

        hidden_size = self.model.config.hidden_size
        for kw in keywords:
            titles = kw_to_titles.get(kw, [])
            if titles:
                vecs = np.stack([title2embed[t] for t in titles])
                kw2title_embed[kw] = vecs.mean(axis=0)
            else:
                # 无关联题名时用关键词自身嵌入占位（后续 encode_keywords 补充）
                kw2title_embed[kw] = np.zeros(hidden_size)

        logger.info("题名语境嵌入计算完成")
        return kw2title_embed


def save_embeddings(kw2embed: Dict[str, np.ndarray], keywords: List[str], embed_file: str, kw_list_file: str = None) -> None:
    """保存嵌入矩阵与关键词列表"""
    embed_matrix = np.stack([kw2embed[kw] for kw in keywords])
    embed_path = PROCESSED_DIR / embed_file
    np.save(embed_path, embed_matrix)
    logger.info(f"嵌入矩阵已保存: {embed_path}，形状: {embed_matrix.shape}")

    if kw_list_file:
        kw_list_path = PROCESSED_DIR / kw_list_file
        with open(kw_list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(keywords))
        logger.info(f"关键词列表已保存: {kw_list_path}")


def load_embeddings(embed_file: str, kw_list_file: str) -> Dict[str, np.ndarray]:
    """加载嵌入矩阵，返回 {keyword: embedding} 字典"""
    embed_path = PROCESSED_DIR / embed_file
    kw_list_path = PROCESSED_DIR / kw_list_file

    embed_matrix = np.load(embed_path)
    with open(kw_list_path, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f.readlines()]

    kw2embed = {kw: embed_matrix[i] for i, kw in enumerate(keywords)}
    logger.info(f"嵌入已加载: {embed_path}，共 {len(kw2embed)} 个关键词")
    return kw2embed


if __name__ == "__main__":
    from src.01_preprocessing.keyword_extractor import filter_valid_keywords
    import pickle

    logger.info("=== BERT 语义嵌入 ===")

    # 加载关键词列表
    kw_list_path = PROCESSED_DIR / "valid_keywords.txt"
    with open(kw_list_path, "r", encoding="utf-8") as f:
        keywords = [line.strip() for line in f.readlines()]
    logger.info(f"加载关键词 {len(keywords)} 个")

    # 加载文献数据（用于题名语境）
    df = pd.read_csv(PROCESSED_DIR / "wos_with_keywords.csv", encoding="utf-8-sig")
    df["keywords"] = df["keywords"].apply(eval)  # 字符串列表转换

    embedder = BERTEmbedder()

    # 1. 关键词文本嵌入
    kw2embed = embedder.encode_keywords(keywords)
    save_embeddings(kw2embed, keywords, "kw_bert_embeddings.npy", "kw_list.txt")

    # 2. 题名语境嵌入
    kw2title_embed = embedder.encode_title_context(keywords, df)
    save_embeddings(
        kw2title_embed, keywords,
        "kw_title_embeddings.npy",
        "kw_list.txt",
    )

    logger.info("BERT 语义嵌入完成！")
