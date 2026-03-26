"""
PRISM Data Processor
Supports multiple prediction modes: total, priority, organization
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple


class GPUDataProcessor:
    """
    Process GPU workload data for different prediction modes
    """

    def __init__(self, node_df: pd.DataFrame, job_df: pd.DataFrame):
        self.node_df = node_df
        self.job_df = job_df

        # Check if priority and organization columns exist
        self.has_priority = 'priority' in job_df.columns or 'job_priority' in job_df.columns
        self.has_organization = 'organization' in job_df.columns or 'user_group' in job_df.columns

    def create_timeseries_total(self, time_window: int = 3600) -> Dict:
        """
        Create total GPU demand time series

        Args:
            time_window: Time window in seconds

        Returns:
            Dictionary with 'total' demand and 'timestamps'
        """
        self.job_df['end_time'] = self.job_df['submit_time'] + self.job_df['duration']
        max_time = self.job_df['end_time'].max()
        time_bins = np.arange(0, max_time + time_window, time_window)

        gpu_demand = []

        for i in range(len(time_bins) - 1):
            start, end = time_bins[i], time_bins[i + 1]
            active_jobs = self.job_df[
                (self.job_df['submit_time'] < end) & (self.job_df['end_time'] > start)
                ]
            total_gpus = (active_jobs['gpu_request'] * active_jobs['worker_num']).sum()
            gpu_demand.append(total_gpus)

        return {
            'total': np.array(gpu_demand),
            'timestamps': time_bins[:-1]
        }

    def create_timeseries_priority(self, time_window: int = 3600) -> Dict:
        """
        Create GPU demand time series by priority (High-Priority vs Spot)

        Args:
            time_window: Time window in seconds

        Returns:
            Dictionary with 'hp' (high-priority), 'spot', 'total', and 'timestamps'
        """
        if not self.has_priority:
            print("Warning: Priority column not found. Falling back to total demand.")
            result = self.create_timeseries_total(time_window)
            result['hp'] = result['total'] * 0.6  # Estimate
            result['spot'] = result['total'] * 0.4
            return result

        # Determine priority column name
        priority_col = 'priority' if 'priority' in self.job_df.columns else 'job_priority'

        self.job_df['end_time'] = self.job_df['submit_time'] + self.job_df['duration']
        max_time = self.job_df['end_time'].max()
        time_bins = np.arange(0, max_time + time_window, time_window)

        hp_demand = []
        spot_demand = []

        for i in range(len(time_bins) - 1):
            start, end = time_bins[i], time_bins[i + 1]
            active_jobs = self.job_df[
                (self.job_df['submit_time'] < end) & (self.job_df['end_time'] > start)
                ]

            # Separate by priority (assuming HP=1, Spot=0 or similar)
            hp_jobs = active_jobs[active_jobs[priority_col] > 0]
            spot_jobs = active_jobs[active_jobs[priority_col] == 0]

            hp_gpus = (hp_jobs['gpu_request'] * hp_jobs['worker_num']).sum()
            spot_gpus = (spot_jobs['gpu_request'] * spot_jobs['worker_num']).sum()

            hp_demand.append(hp_gpus)
            spot_demand.append(spot_gpus)

        hp_demand = np.array(hp_demand)
        spot_demand = np.array(spot_demand)

        return {
            'hp': hp_demand,
            'spot': spot_demand,
            'total': hp_demand + spot_demand,
            'timestamps': time_bins[:-1]
        }

    def create_timeseries_organization(self, time_window: int = 3600,
                                       top_n: int = 5) -> Dict:
        """
        Create GPU demand time series by organization

        Args:
            time_window: Time window in seconds
            top_n: Number of top organizations to track

        Returns:
            Dictionary with demands for each organization and 'timestamps'
        """
        if not self.has_organization:
            print("Warning: Organization column not found. Falling back to total demand.")
            result = self.create_timeseries_total(time_window)
            # Create dummy organizations
            for i in range(top_n):
                result[f'org_{i}'] = result['total'] / top_n
            return result

        # Determine organization column name
        org_col = 'organization' if 'organization' in self.job_df.columns else 'user_group'

        # Find top N organizations by total GPU usage
        # NOTE: Don't multiply groupby objects directly (pandas doesn't support it).
        # Instead, compute per-job total GPU demand first, then aggregate by organization.
        tmp = self.job_df.assign(_gpu_total=self.job_df['gpu_request'] * self.job_df['worker_num'])
        org_gpu_usage = tmp.groupby(org_col)['_gpu_total'].sum()
        top_orgs = org_gpu_usage.nlargest(top_n).index.tolist()

        self.job_df['end_time'] = self.job_df['submit_time'] + self.job_df['duration']
        max_time = self.job_df['end_time'].max()
        time_bins = np.arange(0, max_time + time_window, time_window)

        org_demands = {org: [] for org in top_orgs}
        other_demand = []

        for i in range(len(time_bins) - 1):
            start, end = time_bins[i], time_bins[i + 1]
            active_jobs = self.job_df[
                (self.job_df['submit_time'] < end) & (self.job_df['end_time'] > start)
                ]

            other_gpus = 0
            for org in top_orgs:
                org_jobs = active_jobs[active_jobs[org_col] == org]
                org_gpus = (org_jobs['gpu_request'] * org_jobs['worker_num']).sum()
                org_demands[org].append(org_gpus)

            # Other organizations
            other_jobs = active_jobs[~active_jobs[org_col].isin(top_orgs)]
            other_gpus = (other_jobs['gpu_request'] * other_jobs['worker_num']).sum()
            other_demand.append(other_gpus)

        result = {
            org: np.array(demands) for org, demands in org_demands.items()
        }
        result['other'] = np.array(other_demand)
        result['total'] = sum(result[k] for k in result.keys())
        result['timestamps'] = time_bins[:-1]

        return result

    def create_timeseries(self, mode: str = 'total', time_window: int = 3600,
                          **kwargs) -> Dict:
        """
        Create time series based on prediction mode

        Args:
            mode: 'total', 'priority', or 'organization'
            time_window: Time window in seconds
            **kwargs: Additional arguments for specific modes

        Returns:
            Dictionary with time series data
        """
        if mode == 'total':
            return self.create_timeseries_total(time_window)
        elif mode == 'priority':
            return self.create_timeseries_priority(time_window)
        elif mode == 'organization':
            top_n = kwargs.get('top_n', 5)
            return self.create_timeseries_organization(time_window, top_n)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class GPUDemandDataset(Dataset):
    """
    GPU demand dataset supporting multiple output channels
    """

    def __init__(self, data: np.ndarray, seq_len: int = 96, pred_len: int = 24,
                 mode: str = 'total', time_window_seconds: int = 3600):
        """
        Args:
            data: Time series data (can be 1D or 2D for multi-channel)
            seq_len: Input sequence length
            pred_len: Prediction length
            mode: Prediction mode
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.mode = mode
        self.time_window_seconds = time_window_seconds

        # Handle both 1D and 2D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.n_channels = data.shape[1]
        self.data = data

        # Standardization (per channel)
        self.scalers = []
        self.data_scaled = np.zeros_like(data)

        for i in range(self.n_channels):
            scaler = StandardScaler()
            self.data_scaled[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
            self.scalers.append(scaler)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # Input sequence (all channels)
        x = self.data_scaled[idx:idx + self.seq_len, :]  # [seq_len, n_channels]

        # Output sequence (all channels)
        y = self.data_scaled[idx + self.seq_len:idx + self.seq_len + self.pred_len, :]

        # Time features
        # Derive periodic calendar features from the relative time index.
        # (Paper uses timestamps for temporal embedding; here we map discretized bins to hour/day/month.)
        start_ts = idx * self.time_window_seconds
        ts = [start_ts + i * self.time_window_seconds for i in range(self.seq_len)]
        hours = torch.LongTensor([(t // 3600) % 24 for t in ts])
        days = torch.LongTensor([(t // (24 * 3600)) % 7 for t in ts])
        months = torch.LongTensor([(t // (30 * 24 * 3600)) % 12 for t in ts])
        is_weekend = torch.LongTensor([1 if int(d) >= 5 else 0 for d in days])

        return (
            torch.FloatTensor(x),  # [seq_len, n_channels]
            hours,
            days,
            months,
            is_weekend,
            torch.FloatTensor(y)  # [pred_len, n_channels]
        )

    def inverse_transform(self, data: np.ndarray, channel: int = 0) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale

        Args:
            data: Normalized data
            channel: Which channel to inverse transform

        Returns:
            Original scale data
        """
        if channel >= len(self.scalers):
            channel = 0

        if data.ndim == 1:
            return self.scalers[channel].inverse_transform(data.reshape(-1, 1)).flatten()
        else:
            return self.scalers[channel].inverse_transform(data)


def prepare_datasets(timeseries_data: Dict, seq_len: int, pred_len: int,
                     mode: str) -> Tuple[np.ndarray, List[str]]:
    """
    Prepare data arrays for different prediction modes

    Args:
        timeseries_data: Dictionary with time series data
        seq_len: Input sequence length
        pred_len: Prediction length
        mode: Prediction mode

    Returns:
        data_array: numpy array of shape [time_steps, n_channels]
        channel_names: List of channel names
    """
    if mode == 'total':
        data_array = timeseries_data['total'].reshape(-1, 1)
        channel_names = ['total']

    elif mode == 'priority':
        data_array = np.stack([
            timeseries_data['hp'],
            timeseries_data['spot']
        ], axis=1)
        channel_names = ['hp', 'spot']

    elif mode == 'organization':
        # Get all organization columns (exclude 'timestamps' and 'total')
        org_keys = [k for k in timeseries_data.keys()
                    if k not in ['timestamps', 'total']]
        data_array = np.stack([timeseries_data[k] for k in org_keys], axis=1)
        channel_names = org_keys

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return data_array, channel_names


if __name__ == "__main__":
    # Test data processor
    print("Testing GPUDataProcessor...")

    # Create dummy data
    node_df = pd.DataFrame({
        'node_id': range(10)
    })

    job_df = pd.DataFrame({
        'submit_time': np.random.randint(0, 100000, 1000),
        'duration': np.random.randint(1000, 10000, 1000),
        'gpu_request': np.random.randint(1, 8, 1000),
        'worker_num': np.random.randint(1, 4, 1000),
        'priority': np.random.randint(0, 2, 1000),
        'organization': np.random.choice(['org_a', 'org_b', 'org_c', 'org_d'], 1000)
    })

    processor = GPUDataProcessor(node_df, job_df)

    # Test different modes
    for mode in ['total', 'priority', 'organization']:
        print(f"\nTesting mode: {mode}")
        data = processor.create_timeseries(mode=mode)
        print(f"Keys: {data.keys()}")
        print(f"Total length: {len(data['total'])}")