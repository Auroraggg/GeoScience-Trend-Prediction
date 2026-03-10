"""
BERT / SciBERT 语义嵌入模块
对关键词文本及其题名语境进行批量语义编码，生成高质量节点语义特征
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModel

from src.utils.config import BERT_CONFIG, PROCESSED_DIR
from src.utils.logger import logger


class BertEmbedder:
    """
    BERT/SciBERT 批量语义嵌入器
    支持关键词文本嵌入与题名语境平均嵌入
    """

    def __init__(self, model_name: str = None, device: str = None, batch_size: int = None):
        self.model_name = model_name or BERT_CONFIG["model_name"]
        self.batch_size = batch_size or BERT_CONFIG["batch_size"]
        self.max_length = BERT_CONFIG["max_length"]

        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"加载 BERT 模型: {self.model_name}，设备: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("BERT 模型加载完成")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本，返回 [CLS] token 的嵌入向量

        Args:
            texts: 文本列表

        Returns:
            嵌入矩阵，shape: (N, hidden_size)
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
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

            # 取 [CLS] token 表示
            cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

            if (i // self.batch_size + 1) % 10 == 0:
                logger.info(
                    f"  已处理 {min(i + self.batch_size, len(texts))}/{len(texts)} 条文本"
                )

        return np.vstack(all_embeddings)

    def encode_keywords(self, keywords: List[str]) -> Dict[str, np.ndarray]:
        """
        对关键词列表进行语义嵌入

        Args:
            keywords: 关键词列表

        Returns:
            {keyword: embedding_vector} 字典
        """
        logger.info(f"开始编码 {len(keywords)} 个关键词...")
        embeddings = self.encode(keywords)
        kw2embed = {kw: embeddings[i] for i, kw in enumerate(keywords)}
        logger.info("关键词语义嵌入完成")
        return kw2embed

    def encode_titles_for_keywords(self, df: pd.DataFrame, valid_kws: set) -> Dict[str, np.ndarray]:
        """
        对每个关键词，聚合其所在文献题名，计算平均语义嵌入（题名语境向量）

        Args:
            df: 含 'keywords' 和 'TI' 列的文献 DataFrame
            valid_kws: 有效关键词集合

        Returns:
            {keyword: mean_title_embedding} 字典
        """
        logger.info("构建关键词→题名语境映射...")

        # 建立关��词→题名列表的映射
        kw_to_titles: Dict[str, List[str]] = {kw: [] for kw in valid_kws}
        for _, row in df.iterrows():
            title = str(row.get("TI", "")).strip()
            if not title:
                continue
            for kw in row.get("keywords", []):
                if kw in kw_to_titles:
                    kw_to_titles[kw].append(title)

        # 对每个关键词计算题名平均嵌入
        kw_list = list(valid_kws)
        # 构造代表性文本：取前5条题名拼接（控制长度）
        representative_texts = []
        for kw in kw_list:
            titles = kw_to_titles[kw][:5]
            if titles:
                text = " [SEP] ".join(titles)
            else:
                text = kw  # 若无题名，退化为关键词本身
            representative_texts.append(text)

        logger.info(f"开始编码 {len(representative_texts)} 个关键词的题名语境...")
        embeddings = self.encode(representative_texts)
        title_embed_dict = {kw: embeddings[i] for i, kw in enumerate(kw_list)}
        logger.info("题名语境嵌入完成")
        return title_embed_dict


def save_embeddings( kw2embed: Dict[str, np.ndarray], keywords: List[str], embed_file: str = "kw_bert_embeddings.npy", kw_file: str = "kw_list.txt", ) -> None: 
    """保存嵌入矩阵与关键词列表""" 
    embed_matrix = np.array([kw2embed[kw] for kw in keywords]) 
    embed_path = PROCESSED_DIR / embed_file 
    np.save(embed_path, embed_matrix) 
    logger.info(f"BERT 嵌入矩阵已保存: {embed_path}, shape: {embed_matrix.shape}") 

    kw_path = PROCESSED_DIR / kw_file 
    with open(kw_path, "w", encoding="utf-8") as f: 
        f.write("\n".join(keywords)) 
    logger.info(f"关键词列表已保存: {kw_path}")

def load_embeddings( embed_file: str = "kw_bert_embeddings.npy", kw_file: str = "kw_list.txt", ) -> Dict[str, np.ndarray]: 
    """加载已保存的嵌入""" 
    embed_path = PROCESSED_DIR / embed_file 
    kw_path = PROCESSED_DIR / kw_file 

    embed_matrix = np.load(embed_path) 
    with open(kw_path, "r", encoding="utf-8") as f: 
        keywords = [line.strip() for line in f.readlines()] 

    kw2embed = {kw: embed_matrix[i] for i, kw in enumerate(keywords)} 
    logger.info(f"加载嵌入完成: {len(kw2embed)} 个关键词, 维度: {embed_matrix.shape[1]}") 
    return kw2embed

if __name__ == "__main__": 
    from src.utils.config import PROCESSED_DIR

    # 加载有效关键词
    kw_path = PROCESSED_DIR / "valid_keywords.txt"
    with open(kw_path, "r", encoding="utf-8") as f: 
        valid_kws = [line.strip() for line in f.readlines()] 

    embedder = BertEmbedder()

    # 关键词嵌入
    kw2embed = embedder.encode_keywords(valid_kws) 
    save_embeddings(kw2embed, valid_kws, "kw_bert_embeddings.npy", "kw_list.txt")

    # 题名语境嵌入
    df = pd.read_csv(PROCESSED_DIR / "wos_with_keywords.csv") 
    df["keywords"] = df["keywords"].apply(eval) 
    title_embeds = embedder.encode_titles_for_keywords(df, set(valid_kws))
    save_embeddings(title_embeds, valid_kws, "kw_title_embeddings.npy", "kw_list.txt")

    logger.info("BERT 嵌入全部完成！")
