import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sim.model import EventType


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


def plot_execution_times(results_default: pd.DataFrame, results_skippy: pd.DataFrame):
    results_default['scheduler'] = 'default'
    results_skippy['scheduler'] = 'skippy'
    results_combined = pd.concat([results_default, results_skippy])

    # Only take the POD_SCHEDULED events
    results_combined = results_combined.loc[results_combined['event'].isin([EventType.POD_SCHEDULED])]
    # Convert the podname to an int (to allow proper sorting)
    results_combined['id'] = results_combined['value'].str[4:].astype(int)
    results_combined['image'] = results_combined['additional_attributes'].apply(lambda x: x.get('image'))
    results_combined['execution_time'] = results_combined['additional_attributes'].apply(
        lambda x: float(x.get('execution_time')))

    results_combined = results_combined[['image', 'scheduler', 'execution_time']].groupby('image')

    bp = results_combined.boxplot(by='scheduler', column='execution_time', layout=(1,3), figsize=(8,4))
    [ax_tmp.set_xlabel('') for ax_tmp in np.asarray(bp).reshape(-1)]
    fig = np.asarray(bp).reshape(-1)[0].get_figure()
    fig.suptitle('Execution Times', y=1)
    plt.savefig(f'results/sim_execution_time_boxplot.png')
    plt.show()


def plot_placement_times(results_default: pd.DataFrame, results_skippy: pd.DataFrame):
    results_default['scheduler'] = 'default'
    results_skippy['scheduler'] = 'skippy'
    results_combined = pd.concat([results_default, results_skippy])

    # Only take the POD_SCHEDULED events
    results_combined = results_combined.loc[results_combined['event'].isin([EventType.POD_SCHEDULED])]
    # Convert the podname to an int (to allow proper sorting)
    results_combined['id'] = results_combined['value'].str[4:].astype(int)
    results_combined['image'] = results_combined['additional_attributes'].apply(lambda x: x.get('image'))
    results_combined['placement_time'] = results_combined['additional_attributes'].apply(
        lambda x: float(x.get('execution_time')))

    results_combined = results_combined[['image', 'scheduler', 'placement_time']].groupby('image')

    bp = results_combined.boxplot(by='scheduler', column='placement_time', layout=(1,3), figsize=(8,4))
    [ax_tmp.set_xlabel('') for ax_tmp in np.asarray(bp).reshape(-1)]
    fig = np.asarray(bp).reshape(-1)[0].get_figure()
    fig.suptitle('Placement Times', y=1)
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
    results_combined['tct'] = results_combined['additional_attributes'].apply(
        lambda x: float(x.get('execution_time')) + float(x.get('placement_time')))

    results_combined = results_combined[['image', 'scheduler', 'tct']].groupby('image')

    bp = results_combined.boxplot(by='scheduler', column='tct', layout=(1,3), figsize=(8,4))
    [ax_tmp.set_xlabel('') for ax_tmp in np.asarray(bp).reshape(-1)]
    fig = np.asarray(bp).reshape(-1)[0].get_figure()
    fig.suptitle('Task Completion Times', y=1)
    plt.savefig(f'results/sim_task_completion_time_boxplot.png')
    plt.show()