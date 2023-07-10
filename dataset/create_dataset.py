import os
import pandas as pd
import talib
from talib import abstract
import torch
import numpy as np
from typing import Dict
from datetime import datetime, timedelta

# 1. 读取数据
def load_data(file: str) -> pd.DataFrame:
    folder = './data/csv/'
    df = pd.read_csv(folder + file, index_col='index')
    return df

# 2. 在某一周期过滤某一时间点之后的数据
def filter_data(df: pd.DataFrame, timestamp: int) -> pd.DataFrame:
    return df[df.index <= timestamp]

# 3. 生成 TA-Lib 指标

def generate_indicators(df: pd.DataFrame) -> list[float]:
    df = df.astype('float')
    filtered_df = df[-1000:]  # 仅保留后面的1000条数据

    indicators = []
    ta_list = talib.get_functions()

    # 迴圈執行，看看結果吧！
    for x in ta_list:
        try:
            # x 為技術指標的代碼，透過迴圈填入，再透過 eval 計算出 output
            output = eval('abstract.'+x+'(filtered_df)')
            last_output = output.values[-10:]
            # last_output has nan value
            if np.isnan(last_output).any():
                continue
            # last_output's element is array or float
            if isinstance(last_output[0], np.ndarray):
                last_output = [item for sublist in last_output for item in sublist]
            indicators.extend(last_output)
        except:
            print(x)
    return indicators


# 4. 将所有周期的指标合并为一个数组
def combine_indicators(indicators_list: list) -> np.array:
    return np.concatenate(indicators_list)

# 5. 将数据保存为 Tensor
def save_as_tensor(data: np.array, timestamp: int):
    tensor = torch.from_numpy(data)
    torch.save(tensor, f'{timestamp}.pt')


def main():
    start_time = datetime(2021, 1, 1)
    end_time = datetime.now()

    total_seconds = int((end_time - start_time).total_seconds())

    timeframes = ["1m", "1h", "1d"]

    all_indicators = []  # 一维数组用于存储所有指标的值

    for sec in range(total_seconds):
        timestamp = int((start_time + timedelta(seconds=sec)).timestamp())
        indicators_per_sec = []  # 存储每个时间戳的指标值

        for timeframe in timeframes:
            # Load and filter data
            df = load_data(f'{timeframe}.csv')
            df = filter_data(df, timestamp)

            # Generate indicators
            indicators = generate_indicators(df)
            indicators_per_sec.extend(indicators)

        all_indicators.extend(indicators_per_sec)
        # Save indicators as a one-dimensional array
        combined_indicators = np.array(all_indicators)
        save_as_tensor(combined_indicators, timestamp)


if __name__ == "__main__":
    main()
