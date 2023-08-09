import torch
from torch.utils.data import Dataset
import h5py

class MyDataset(Dataset):
    
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        
        feature_file = './data/dataset/feature.h5' 
        label_file = './data/dataset/label.h5'
        
        self.features = h5py.File(feature_file, 'r')
        self.labels = h5py.File(label_file, 'r')
        
        self.timestamps = []
        for key in self.features.keys():
            timestamp = int(key)
            if timestamp >= start_time and timestamp <= end_time:
                self.timestamps.append(timestamp)
        
    def __len__(self):
        return len(self.timestamps)
    
    def __getitem__(self, idx):
        timestamp = str(self.timestamps[idx])
        
        features = torch.tensor(self.features[timestamp][:]) # type: ignore
        labels = torch.tensor(self.labels[timestamp][:]) # type: ignore
        
        return features, labels