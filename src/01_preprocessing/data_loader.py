"""
WoS 数据加载与清洗模块
支持 Web of Science 制表符分隔 .txt 和 .csv 格式导入
"""

import re
import glob
import pandas as pd
from pathlib import Path
from typing import Optional

from src.utils.config import RAW_DIR, PROCESSED_DIR, PREPROCESS_CONFIG
from src.utils.logger import logger


# WoS 标准字段映射
WOS_FIELDS = {
    "TI": "title",
    "AB": "abstract",
    "DE": "author_keywords",
    "ID": "keywords_plus",
    "SC": "subject_category",
    "PY": "year",
    "UT": "accession_number",
    "SO": "journal",
    "AU": "authors",
    "C1": "affiliations",
}

def load_wos_data(
    data_dir: Path = None,
    encoding: str = None,
    sep: str = None,
) -> pd.DataFrame:
    """
    加载 WoS 导出数据，支持合并多个导出文件

    Args:
        data_dir: 数据目录，默认使用 config 中的 RAW_DIR
        encoding: 文件编码
        sep: 字段分隔符

    Returns:
        合并后的原始 DataFrame（字段使用 WoS 原始代码）
    """
    data_dir = data_dir or RAW_DIR
    encoding = encoding or PREPROCESS_CONFIG["encoding"]
    sep = sep or PREPROCESS_CONFIG["sep"]

    # 搜索所有 txt 和 csv 文件
    txt_files = list(data_dir.glob("*.txt"))
    csv_files = list(data_dir.glob("*.csv"))
    all_files = txt_files + csv_files

    if not all_files:
        raise FileNotFoundError(
            f"在 {data_dir} 中未找到任何 .txt 或 .csv 文件\n"
            "请将 WoS 导出文件放入 data/raw/ 目录"
        )

    logger.info(f"发现 {len(all_files)} 个数据文件，开始加载...")
    dfs = []
    for f in all_files:
        try:
            if f.suffix == ".txt":
                # WoS .txt 格式：制表符分隔，前两行为标识行
                df = pd.read_csv(f, sep=sep, encoding=encoding, dtype=str, skiprows=0)
            else:
                df = pd.read_csv(f, encoding=encoding, dtype=str)
            dfs.append(df)
            logger.info(f"  已加载: {f.name}  ({len(df)} 条记录)")
        except Exception as e:
            logger.warning(f"  跳过文件 {f.name}，原因: {e}")

    raw_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"原始数据总量: {len(raw_df)} 条")
    return raw_df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗与字段标准化

    处理步骤：
    1. 重命名字段为英文全名
    2. 过滤年份范围
    3. 去重
    4. 规范化年份类型
    5. 过滤必要字段缺失的记录

    Args:
        df: 原始 WoS DataFrame

    Returns:
        清洗后的 DataFrame
    """
    logger.info("开始数据清洗...")

    # 1. 重命名字段（仅保留存在的字段）
    rename_map = {k: v for k, v in WOS_FIELDS.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    logger.info(f"字段重命名完成，可用字段: {list(df.columns)}")

    # 2. 规范化年份
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        year_start = PREPROCESS_CONFIG["year_start"]
        year_end = PREPROCESS_CONFIG["year_end"]
        before = len(df)
        df = df[df["year"].between(year_start, year_end)]
        logger.info(
            f"年份过滤 ({year_start}–{year_end}): "
            f"{before} → {len(df)} 条"
        )
    else:
        logger.warning("未找到年份字段 'PY'，跳过年份过滤")

    # 3. 去重（基于唯一标识符）
    if "accession_number" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["accession_number"])
        logger.info(f"去重后: {before} → {len(df)} 条")
    else:
        # 基于题名去重
        if "title" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["title"])
            logger.info(f"基于题名去重后: {before} → {len(df)} 条")

    # 4. 过滤关键必要字段
    required_fields = ["title"]
    for field in required_fields:
        if field in df.columns:
            before = len(df)
            df = df[df[field].notna() & (df[field].str.strip() != "")]
            logger.info(f"过滤 {field} 缺失: {before} → {len(df)} 条")

    # 5. 重置索引
    df = df.reset_index(drop=True)
    logger.info(f"数据清洗完成，最终记录数: {len(df)}")

    return df

def get_year_distribution(df: pd.DataFrame) -> pd.Series:
    """返回各年份文献数量统计"""
    if "year" not in df.columns:
        raise ValueError("DataFrame 中无 'year' 字段")
    return df["year"].value_counts().sort_index()

if __name__ == "__main__":
    logger.info("=== WoS 数据加载测试 ===")
    raw_df = load_wos_data()
    clean_df = clean_data(raw_df)

    # 保存清洗后数据
    out_path = PROCESSED_DIR / "wos_cleaned.csv"
    clean_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"清洗后数据已保存: {out_path}")

    # 打印年份分布
    year_dist = get_year_distribution(clean_df)
    logger.info(f"各年份文献数量:\n{year_dist.to_string()}")
