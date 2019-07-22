from typing import Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sim.model import EventType


def plot_combined_placement_time_cdf(dfs: List[Tuple[str, pd.DataFrame]]):
    for scheduler_name, df_1 in dfs:
        # Only take the POD_QUEUED and the pod_scheduled events
        df_1 = df_1.loc[df_1['event'].isin([EventType.POD_RECEIVED, EventType.POD_SCHEDULED])]
        # Filter pod events of pods which have not fully been scheduled (not all 3 events are included)
        df_1 = df_1.groupby(['value']).filter(lambda x: len(x) == 2)
        # Convert the podname to an int (to allow proper sorting)
        df_1['value'] = df_1['value'].str[4:].astype(int)
        # Sort by pods, then by event (POD_QUEUED < POD_SCHEDULED)
        df_1 = df_1.sort_values(['value', 'event'], ascending=[True, True])
        # Drop the diff between two pods (every second entry)
        ser = df_1['timestamp'].diff().iloc[1::2]
        # Adopt the index
        ser.index = range(len(ser))
        # Transform from seconds to milliseconds
        ser = ser * 1000

        # Create the CDF of the series and plot it
        x, y = sorted(ser), np.arange(len(ser)) / len(ser)
        plt.plot(x, y, label=f'placement time using the {scheduler_name} scheduler')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Task Placement Latency (ms)')
    plt.savefig(f'results/sim_placement_time_cdf_combined.png')
    plt.show()


def plot_placement_time_cdf(df_1: pd.DataFrame, scheduler_name: str):
    # Only take the POD_QUEUED and the pod_scheduled events
    df_1 = df_1.loc[df_1['event'].isin([EventType.POD_RECEIVED, EventType.POD_SCHEDULED])]
    # Filter pod events of pods which have not fully been scheduled (not all 3 events are included)
    df_1 = df_1.groupby(['value']).filter(lambda x: len(x) == 2)
    # Convert the podname to an int (to allow proper sorting)
    df_1['value'] = df_1['value'].str[4:].astype(int)
    # Sort by pods, then by event (POD_QUEUED < POD_SCHEDULED)
    df_1 = df_1.sort_values(['value', 'event'], ascending=[True, True])
    # Drop the diff between two pods (every second entry)
    ser = df_1['timestamp'].diff().iloc[1::2]
    # Adopt the index
    ser.index = range(len(ser))
    # Transform from seconds to milliseconds
    ser = ser * 1000

    # Create the CDF of the series and plot it
    x, y = sorted(ser), np.arange(len(ser)) / len(ser)
    plt.plot(x, y, label=f'placement time using the {scheduler_name} scheduler')

    plt.legend()
    plt.ylabel('Probability')
    plt.xlabel('Task Placement Latency (ms)')
    plt.savefig(f'results/sim_{scheduler_name}_placement_time_cdf.png')
    plt.show()


def plot_execution_times_boxplot(results_default: pd.DataFrame, results_skippy: pd.DataFrame):
    results_default['scheduler'] = 'default'
    results_skippy['scheduler'] = 'skippy'
    results_combined = pd.concat([results_default, results_skippy])

    # Only take the POD_SCHEDULED events
    results_combined = results_combined.loc[results_combined['event'].isin([EventType.POD_SCHEDULED])]
    # Convert the podname to an int (to allow proper sorting)
    results_combined['id'] = results_combined['value'].str[4:].astype(int)
    results_combined['image'] = results_combined['additional_attributes'].apply(lambda x: x.get('image'))

    # Remove the serving-images, exec time is negligible for them
    results_combined = results_combined.loc[~results_combined['image'].isin(['alexrashed/ml-wf-3-serve:0.33'])]

    # Extract the execution time from the additional attributes
    results_combined['execution_time'] = results_combined['additional_attributes'].apply(
        lambda x: float(x.get('execution_time')))

    # Extract the execution time from the additional attributes
    results_combined = results_combined[['image', 'scheduler', 'execution_time']].groupby('image')

    bp = results_combined.boxplot(by='scheduler', column='execution_time', layout=(1,2), figsize=(8,4))
    [ax_tmp.set_xlabel('') for ax_tmp in np.asarray(bp).reshape(-1)]
    fig = np.asarray(bp).reshape(-1)[0].get_figure()
    fig.suptitle('Execution Times (s)', y=1)
    plt.savefig(f'results/sim_execution_time_boxplot.png')
    plt.show()


def plot_execution_times_bar(results_default: pd.DataFrame, results_skippy: pd.DataFrame):
    results_default['scheduler'] = 'default'
    results_skippy['scheduler'] = 'skippy'
    results_combined = pd.concat([results_default, results_skippy])

    # Only take the POD_SCHEDULED events
    results_combined = results_combined.loc[results_combined['event'].isin([EventType.POD_SCHEDULED])]
    # Convert the podname to an int (to allow proper sorting)
    results_combined['id'] = results_combined['value'].str[4:].astype(int)
    results_combined['image'] = results_combined['additional_attributes'].apply(lambda x: x.get('image'))

    # Remove the serving-images, exec time is negligible for them
    results_combined = results_combined.loc[~results_combined['image'].isin(['alexrashed/ml-wf-3-serve:0.33'])]

    # Extract the execution time from the additional attributes
    results_combined['execution_time'] = results_combined['additional_attributes'].apply(
        lambda x: float(x.get('execution_time')))

    grouped = results_combined[['image', 'scheduler', 'execution_time']].groupby(['scheduler', 'image'])
    mean = grouped.execution_time.mean().unstack(0)
    ax = mean.plot.bar()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
    plt.ylabel('Execution Times (s)')
    plt.xlabel('Images')
    plt.savefig(f'results/sim_placement_time_cdf_combined.png')
    plt.savefig(f'results/sim_execution_time_bar.png')
    plt.show()


def plot_placement_times_bar(results_default: pd.DataFrame, results_skippy: pd.DataFrame):
    results_default['scheduler'] = 'default'
    results_skippy['scheduler'] = 'skippy'
    results_combined = pd.concat([results_default, results_skippy])

    # Only take the POD_SCHEDULED events
    results_combined = results_combined.loc[results_combined['event'].isin([EventType.POD_SCHEDULED])]
    # Convert the podname to an int (to allow proper sorting)
    results_combined['id'] = results_combined['value'].str[4:].astype(int)
    results_combined['image'] = results_combined['additional_attributes'].apply(lambda x: x.get('image'))
    results_combined['placement_time'] = results_combined['additional_attributes'].apply(
        lambda x: float(x.get('placement_time')))

    grouped = results_combined[['image', 'scheduler', 'placement_time']].groupby(['scheduler', 'image'])
    mean = grouped.placement_time.mean().unstack(0)
    ax = mean.plot.bar()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=10)
    plt.ylabel('Placement Times (s)')
    plt.xlabel('Images')
    plt.savefig(f'results/sim_placement_time_bar.png')
    plt.show()


def plot_placement_times_boxplot(results_default: pd.DataFrame, results_skippy: pd.DataFrame):
    results_default['scheduler'] = 'default'
    results_skippy['scheduler'] = 'skippy'
    results_combined = pd.concat([results_default, results_skippy])

    # Only take the POD_SCHEDULED events
    results_combined = results_combined.loc[results_combined['event'].isin([EventType.POD_SCHEDULED])]
    # Convert the podname to an int (to allow proper sorting)
    results_combined['id'] = results_combined['value'].str[4:].astype(int)
    results_combined['image'] = results_combined['additional_attributes'].apply(lambda x: x.get('image'))
    results_combined['placement_time'] = results_combined['additional_attributes'].apply(
        lambda x: float(x.get('placement_time')))

    results_combined = results_combined[['image', 'scheduler', 'placement_time']].groupby('image')

    bp = results_combined.boxplot(by='scheduler', column='placement_time', layout=(1,3), figsize=(8,4))
    [ax_tmp.set_xlabel('') for ax_tmp in np.asarray(bp).reshape(-1)]
    fig = np.asarray(bp).reshape(-1)[0].get_figure()
    fig.suptitle('Placement Times (s)', y=1)
    plt.savefig(f'results/sim_placement_time_boxplot.png')
    plt.show()


def plot_task_completion_times(results_default: pd.DataFrame, results_skippy: pd.DataFrame):
    results_default['scheduler'] = 'default'
    results_skippy['scheduler'] = 'skippy'
    results_combined = pd.concat([results_default, results_skippy])

    # Only take the POD_SCHEDULED events
    results_combined = results_combined.loc[results_combined['event'].isin([EventType.POD_SCHEDULED])]
    # Convert the podname to an int (to allow proper sorting)
    results_combined['id'] = results_combined['value'].str[4:].astype(int)
    results_combined['image'] = results_combined['additional_attributes'].apply(lambda x: x.get('image'))

    # Remove the serving-images, exec time is negligible for them
    results_combined = results_combined.loc[~results_combined['image'].isin(['alexrashed/ml-wf-3-serve:0.33'])]

    results_combined['tct'] = results_combined['additional_attributes'].apply(
        lambda x: float(x.get('execution_time')) + float(x.get('placement_time')))

    results_combined = results_combined[['image', 'scheduler', 'tct']].groupby('image')

    bp = results_combined.boxplot(by='scheduler', column='tct', layout=(1,2), figsize=(8,4))
    [ax_tmp.set_xlabel('') for ax_tmp in np.asarray(bp).reshape(-1)]
    fig = np.asarray(bp).reshape(-1)[0].get_figure()
    fig.suptitle('Task Completion Times (s)', y=1)
    plt.savefig(f'results/sim_task_completion_time_boxplot.png')
    plt.show()