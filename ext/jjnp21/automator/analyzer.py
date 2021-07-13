from typing import List, Dict

from pandas import DataFrame

from ext.jjnp21.automator.experiment import Result


def set_generic_kpis(row: Dict, df: DataFrame, name: str, property_key: str):
    row[name + ' mean'] = round(df[property_key].mean(), 4)
    row[name + ' q50'] = round(df[property_key].quantile(0.5), 4)
    row[name + ' q75'] = round(df[property_key].quantile(0.75), 4)
    row[name + ' q90'] = round(df[property_key].quantile(0.90), 4)
    row[name + ' q99'] = round(df[property_key].quantile(0.99), 4)


def set_e2e_kpis(row: Dict, inv_df: DataFrame):
    set_generic_kpis(row, inv_df, 'E2E', 't_exec')


def set_fet_kpis(row: Dict, fet_df: DataFrame):
    set_generic_kpis(row, fet_df, 'FET', 't_fet')


def set_fx_wait_kpis(row: Dict, fet_df: DataFrame):
    set_generic_kpis(row, fet_df, 'wait', 't_wait')


def set_total_reuqest_count(row: Dict, inv_df: DataFrame):
    row['total requests'] = inv_df['t_exec'].count()


def set_request_tx_kpis(row: Dict, inv_df: DataFrame):
    # In case we only have dummy values, don't include it
    if inv_df['tx_time_cl_lb'].mean() == 0:
        return
    set_generic_kpis(row, inv_df, 'tx_time_cl_lb', 'tx_time_cl_lb')
    set_generic_kpis(row, inv_df, 'tx_time_lb_fx', 'tx_time_lb_fx')


def set_node_type_distribution(row: Dict, inv_df: DataFrame):
    nodes = inv_df['node'].unique()
    types = ['rpi3', 'rpi4', 'tx2', 'nuc', 'rockpi', 'nx', 'coral', 'nano', 'xeoncpu', 'xeongpu', 'cloudvm']
    results = {}
    node_counts = {}
    typed_results = {}
    avg_typed_results = {}
    for n in nodes:
        cnt = inv_df[inv_df['node'] == n]['node'].count()
        results[n] = cnt
    for t in types:
        typed_results[t] = 0
        node_counts[t] = 0
        for n, cnt in results.items():
            if n.startswith(t):
                typed_results[t] += cnt
                node_counts[t] += 1

    for t, cnt in typed_results.items():
        avg_typed_results[t] = cnt / max(node_counts[t], 1)

    row['nodes_by_type'] = str(node_counts)
    row['total_rq_by_type'] = str(typed_results)
    row['avg_rq_by_type'] = str(avg_typed_results)


class BasicResultAnalyzer:
    def __init__(self, results: List[Result]):
        self.results = results

    def basic_kpis(self, include_node_type_distribution: bool = True) -> DataFrame:
        rows = []
        for result in self.results:
            row = {}
            row['name'] = result.experiment.name
            set_e2e_kpis(row, result.invocations)
            set_fet_kpis(row, result.fets)
            set_fx_wait_kpis(row, result.fets)
            set_total_reuqest_count(row, result.invocations)
            set_request_tx_kpis(row, result.invocations)
            if include_node_type_distribution:
                set_node_type_distribution(row, result.invocations)
            rows.append(row)
        return DataFrame(rows)
