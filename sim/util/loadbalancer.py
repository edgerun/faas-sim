from faas.util.constant import controller_role_label


def find_lbs(topology):
    return [x for x in topology.get_nodes() if x.labels.get(controller_role_label) is not None]