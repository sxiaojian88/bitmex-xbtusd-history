import os
import pandas as pd
import talib
from talib import abstract
import numpy as np
from datetime import datetime, timedelta
import h5py
import torch
from concurrent.futures import ThreadPoolExecutor

class DatasetCreator:
    def __init__(self, csv_data_folder='./data/csv/', dataset_dir='./data/dataset/', feature_file='./data/dataset/feature.h5', label_file='./data/dataset/label.h5'):
        self.csv_data_folder = csv_data_folder
        self.dataset_dir = dataset_dir
        self.feature_file = feature_file
        self.label_file = label_file
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

    def save_features_to_h5(self, timestamp: str, features: list[float]) -> None:
        """Save features to an h5py file."""
        with h5py.File(self.feature_file, 'w') as hf:
            hf.create_dataset(timestamp, data=features)

    def save_labels_to_h5(self, timestamp: str, labels: list[int]) -> None:
        """Save features to an h5py file."""
        with h5py.File(self.label_file, 'w') as hf:
            hf.create_dataset(timestamp, data=labels)

    def load_data(self, file: str) -> pd.DataFrame:
        """Load data from a given file."""
        return pd.read_csv(self.csv_data_folder + file, index_col='index')  # type: ignore

    @staticmethod
    def filter_data(df: pd.DataFrame, timestamp: int) -> pd.DataFrame:
        """Filter data before a certain timestamp."""
        return df[df.index <= timestamp]

    @staticmethod
    def generate_indicators(df: pd.DataFrame) -> list[float]:
        # 只取最后1000个数据
        filtered_df = df[-1000:]
        indicators = []
        for function in talib.get_functions():
            if function == 'MAVP':
                continue
            output = eval(f'abstract.{function}(filtered_df)')
            # 替换NaN
            last_output = np.nan_to_num(output)
            last_output = torch.from_numpy(last_output)
            last_output = last_output.float()
            shape = last_output.shape
            # 检查是否是二维数组
            if len(shape) == 1:
                last_output = last_output.reshape(shape[0], 1)
            x_normalized = torch.nn.functional.normalize(last_output, dim=0)
            x_normalized_list = x_normalized.reshape(-1).tolist()
            latest_indicators = x_normalized_list[-10*last_output.shape[1]:]
            indicators.extend(latest_indicators)

        return indicators

    def process_in_time_range(self, start_time=datetime(2020, 2, 1), end_time=datetime(2023, 7, 1)):
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        with h5py.File(self.label_file, 'w') as hf:
            with ThreadPoolExecutor(max_workers=1) as executor:
                for minute in range(total_minutes):
                    current_time = start_time + timedelta(minutes=minute)
                    timestamp = int(current_time.timestamp())
                    executor.submit(self.compute_features_and_labels, timestamp)

    def compute_features_and_labels(self, timestamp: int):
        # Load data
        df_1m = self.load_data('1m.csv')
        df_1h = self.load_data('1h.csv')
        df_1d = self.load_data('1d.csv')

        # Filter data
        df_1m_filtered = self.filter_data(df_1m, timestamp)
        df_1h_filtered = self.filter_data(df_1h, timestamp)
        df_1d_filtered = self.filter_data(df_1d, timestamp)

        # Generate indicators
        indicators_1m = self.generate_indicators(df_1m_filtered)
        indicators_1h = self.generate_indicators(df_1h_filtered)
        indicators_1d = self.generate_indicators(df_1d_filtered)

        features = indicators_1m + indicators_1h + indicators_1d
        labels = self.generate_labels(timestamp=timestamp, df_1m=df_1m)
        date = datetime.utcfromtimestamp(timestamp)
        print(f"writing date {date}\n")
        self.save_features_to_h5(
            timestamp=str(timestamp), features=features)
        self.save_labels_to_h5(timestamp=str(timestamp), labels=labels)

    def generate_labels(self, timestamp: int, df_1m: pd.DataFrame) -> list[int]:
        """Generate labels for a specific timestamp."""
        earn_value = 5
        loss_value = 40

        df_1m = df_1m.sort_index(ascending=True)

        start_time = datetime.utcfromtimestamp(timestamp)
        end_time = start_time + timedelta(hours=24)
        mask = (df_1m.index >= start_time.timestamp()) & (df_1m.index <= end_time.timestamp())
        next_24h_data = df_1m[mask]
        startPrice = next_24h_data.iloc[0]['open']

        open_values = next_24h_data['open']
        min_open = open_values.min()
        max_open = open_values.max()

        if max_open > startPrice * 1.02 or min_open < startPrice * 0.98:
            return [0, 0]

        label_1 = 0
        label_2 = 0

        for _, row in next_24h_data.iterrows():
            currentPrice = row['open']
            if label_1 == 0:
                if currentPrice - earn_value > startPrice:
                    label_1 = 1
                elif currentPrice + loss_value < startPrice:
                    label_1 = 0
            if label_2 == 0:
                if currentPrice + earn_value < startPrice:
                    label_2 = 1
                elif currentPrice - loss_value > startPrice:
                    label_2 = 0
            if label_1 and label_2:
                break

        return [label_1, label_2]

if __name__ == "__main__":
    # Create a new instance of the class
    data_processor = DatasetCreator()
    data_processor.process_in_time_range()
    # test
    # data_processor = DatasetCreator()
    # data_processor.compute_features_and_labels(int(datetime(2020, 2, 1).timestamp()))
