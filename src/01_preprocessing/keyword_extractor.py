"""
关键词抽取与频次筛选模块
融合作者关键词(DE)与扩展关键词(ID)，基于频次与长度阈值筛选有效关键词
"""

import re
import pandas as pd
from collections import Counter
from pathlib import Path
from typing import List, Set, Dict, Tuple

from src.utils.config import KEYWORD_CONFIG, PROCESSED_DIR
from src.utils.logger import logger


# 通用停用词（可按领域扩充）
STOPWORDS = {
    "and", "of", "in", "the", "a", "an", "for", "to", "on", "with",
    "by", "from", "at", "as", "is", "are", "was", "were", "be",
    "this", "that", "these", "those", "based", "using", "study",
    "analysis", "research", "review", "new", "novel", "approach",
    "method", "model", "data", "results", "effect", "effects",
}

def parse_keywords(kw_str: str) -> List[str]:
    """
    解析单条关键词字段字符串
    WoS 关键词以分号分隔

    Args:
        kw_str: 原始关键词字符串，如 "climate change; machine learning; remote sensing"

    Returns:
        关键词列表（小写、去空格）
    """
    if not kw_str or pd.isna(kw_str):
        return []
    kws = [kw.strip().lower() for kw in str(kw_str).split(";")]  
    # 过滤空字符串
    kws = [kw for kw in kws if kw]
    return kws

def clean_keyword(kw: str) -> str:
    """
    关键词规范化处理
    - 去除首尾空格
    - 合并多余空格
    - 去除特殊字符
    """
    kw = kw.strip().lower()
    kw = re.sub(r"\s+", " ", kw)          # 合并空格
    kw = re.sub(r"[^\w\s\-]", "", kw)     # 保留字母/数字/空格/连字符
    kw = kw.strip()
    return kw

def extract_keywords_from_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Counter]:
    """
    从文献 DataFrame 中提取并规范化关键词

    Args:
        df: 清洗后的文献数据

    Returns:
        (df_with_kw, global_freq_counter)
        - df_with_kw: 增加了 'keywords_raw' 列的 DataFrame
        - global_freq_counter: 全局关键词频次计数器
    """
    logger.info("开始关键词抽取...")

    all_keywords = []

    def extract_row_keywords(row) -> List[str]:
        kws = []
        if KEYWORD_CONFIG["use_author_kw"] and "DE" in row.index:
            kws += parse_keywords(row.get("DE", ""))
        if KEYWORD_CONFIG["use_plus_kw"] and "ID" in row.index:
            kws += parse_keywords(row.get("ID", ""))
        # 规范化
        kws = [clean_keyword(kw) for kw in kws]
        # 去重（同一篇文献内）
        seen = set()
        unique_kws = []
        for kw in kws:
            if kw and kw not in seen:
                seen.add(kw)
                unique_kws.append(kw)
        return unique_kws

    df = df.copy()
    df["keywords_raw"] = df.apply(extract_row_keywords, axis=1)

    # 统计全局频次
    all_kws_flat = [kw for kws in df["keywords_raw"] for kw in kws]
    global_freq = Counter(all_kws_flat)

    logger.info(f"原始关键词种类数: {len(global_freq)}")
    logger.info(f"关键词总出现次数: {sum(global_freq.values())}")

    return df, global_freq

def filter_valid_keywords(
    global_freq: Counter,
    min_freq: int = None,
    min_len: int = None,
    max_len: int = None,
) -> Set[str]:
    """
    基于频次与长度阈值筛选有效关键词

    Args:
        global_freq: 全局频次计数器
        min_freq: 最低频次（默认使用 config）
        min_len: 最短长度
        max_len: 最长长度

    Returns:
        有效关键词集合
    """
    min_freq = min_freq or KEYWORD_CONFIG["min_freq"]
    min_len = min_len or KEYWORD_CONFIG["min_len"]
    max_len = max_len or KEYWORD_CONFIG["max_len"]

    valid = set()
    for kw, freq in global_freq.items():
        if freq < min_freq:
            continue
        if len(kw) < min_len or len(kw) > max_len:
            continue
        if kw in STOPWORDS:
            continue
        valid.add(kw)

    logger.info(
        f"有效关键词筛选完成（min_freq={min_freq}, min_len={min_len}, max_len={max_len}）: "
        f"{len(valid)} 个"
    )
    return valid

def apply_filter_to_df(df: pd.DataFrame, valid_kws: Set[str]) -> pd.DataFrame:
    """
    将有效关键词集合应用于 DataFrame，过滤无效关键词

    Args:
        df: 含 'keywords_raw' 列的 DataFrame
        valid_kws: 有效关键词集合

    Returns:
        含 'keywords' 列（过滤后）的 DataFrame
    """
    df = df.copy()
    df["keywords"] = df["keywords_raw"].apply(
        lambda kws: [kw for kw in kws if kw in valid_kws]
    )
    # 过滤掉关键词数量不足的文献
    before = len(df)
    df = df[df["keywords"].apply(len) >= 2]
    logger.info(
        f"过滤后：保留 {len(df)} 篇文献（删除 {before - len(df)} 篇关键词不足的文献）"
    )
    return df

def save_keywords(
    valid_kws: Set[str],
    global_freq: Counter,
    kw_file: str = "valid_keywords.txt",
    freq_file: str = "keyword_freq.csv",
) -> None:
    """保存有效关键词列表与频次统计"""
    # 保存关键词列表（按频次排序）
    sorted_kws = sorted(valid_kws, key=lambda k: global_freq[k], reverse=True)
    kw_path = PROCESSED_DIR / kw_file
    with open(kw_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted_kws))
    logger.info(f"关键词列表已保存: {kw_path}")

    # 保存频次统计
    freq_df = pd.DataFrame(
        [(kw, global_freq[kw]) for kw in sorted_kws],
        columns=["keyword", "frequency"]
    )
    freq_path = PROCESSED_DIR / freq_file
    freq_df.to_csv(freq_path, index=False, encoding="utf-8-sig")
    logger.info(f"关键词频次已保存: {freq_path}")

if __name__ == "__main__":
    from src.01_preprocessing.data_loader import load_wos_data, clean_data

    logger.info("=== 关键词抽取与筛选 ===")
    raw_df = load_wos_data()
    clean_df = clean_data(raw_df)

    df_with_kw, global_freq = extract_keywords_from_df(clean_df)
    valid_kws = filter_valid_keywords(global_freq)
    df_final = apply_filter_to_df(df_with_kw, valid_kws)

    save_keywords(valid_kws, global_freq)

    # 保存带关键词的文献数据
    df_final.to_csv(PROCESSED_DIR / "wos_with_keywords.csv", index=False, encoding="utf-8-sig")
    logger.info("关键词抽取完成！")
