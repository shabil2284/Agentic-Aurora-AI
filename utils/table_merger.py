import pandas as pd
from typing import List, Dict, Any
import numpy as np

class TableMerger:
    @staticmethod
    def merge_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
        if not tables:
            return pd.DataFrame()
        if len(tables) == 1:
            return tables[0]

        dfs = []
        for df in tables:
            if isinstance(df, pd.DataFrame) and not df.empty:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # Use ignore_index=True to reset the internal pandas integer index only.
        # This does NOT modify the SNo column — SNo values are preserved from the data.
        merged_df = pd.concat(dfs, ignore_index=True)

        # Remove duplicate header rows (where cell value equals column name)
        for col in merged_df.columns:
            merged_df = merged_df[merged_df[col] != col]

        return merged_df

    @staticmethod
    def detect_continuation(page_n: int, page_n_plus_1_blocks: List[Any]) -> bool:
        return False
