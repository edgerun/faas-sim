import os
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd
from faas.context import NodeService
from faas.system import FunctionNode

from sim.faassim import Simulation


def extract_nodes(sim: Simulation):
    service: NodeService[FunctionNode] = sim.env.context.node_service
    data = defaultdict(list)
    keys = ['name', 'arch', 'cpus', 'ram', 'netspeed', 'labels', 'allocatable', 'cluster', 'state']
    for node in service.get_nodes():
        for k in keys:
            data[k].append(node.__dict__[k])

    df = pd.DataFrame(data=data)
    return df


def save_results(root_folder: str, dfs: Dict[str, pd.DataFrame], sim: Simulation):
    exp_id = dfs['experiment_df']['EXP_ID'].iloc[0]
    path = f'{root_folder}/{exp_id}'
    if os.path.exists(path):
        raise ValueError(f'Path {path} already exists. Stop saving results.')
    Path(path).mkdir(parents=True, exist_ok=False)
    for name, df in dfs.items():
        file_name = f'{path}/{name}.csv'
        df.to_csv(file_name)

    return path


def extract_dfs(sim):
    return {
        'allocation_df': sim.env.metrics.extract_dataframe('allocation'),
        'experiment_df': sim.env.metrics.extract_dataframe('experiment'),
        'invocations_df': sim.env.metrics.extract_dataframe('invocations'),
        'traces_df': sim.env.metrics.extract_dataframe('traces'),
        'scale_df': sim.env.metrics.extract_dataframe('scale'),
        'schedule_df': sim.env.metrics.extract_dataframe('schedule'),
        'replica_deployment_df': sim.env.metrics.extract_dataframe('replica_deployment'),
        'function_deployments_df': sim.env.metrics.extract_dataframe('function_deployments'),
        'function_deployment_lifecycle_df': sim.env.metrics.extract_dataframe('function_deployment_lifecycle'),
        'function_containers_df': sim.env.metrics.extract_dataframe('function_containers'),
        'function_images_df': sim.env.metrics.extract_dataframe('function_images'),
        'function_replicas_df': sim.env.metrics.extract_dataframe('function_replicas'),
        'functions_df': sim.env.metrics.extract_dataframe('functions'),
        'flow_df': sim.env.metrics.extract_dataframe('flow'),
        'network_df': sim.env.metrics.extract_dataframe('network'),
        'node_utilization_df': sim.env.metrics.extract_dataframe('node_utilization'),
        'function_utilization_df': sim.env.metrics.extract_dataframe('function_utilization'),
        'fets_df': sim.env.metrics.extract_dataframe('fets'),
        'nodes': extract_nodes(sim)
    }
