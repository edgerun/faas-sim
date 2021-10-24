import pandas as pd
import matplotlib.pyplot as plt

def plot_rtt(df: pd.DataFrame, resampling_mean_interval_seconds: int = 5, initial_cutoff_seconds: int = 0):
    data = df[df['t_start'] >= initial_cutoff_seconds]
    plt.plot(data['t_exec'].resample(f'{resampling_mean_interval_seconds}s').mean())
    plt.show()