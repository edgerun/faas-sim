import time

import psutil
import pandas as pd
from collections import defaultdict
# Calling psutil.cpu_precent() for 5 seconds
data = defaultdict(list)
try:
    while True:
        cpu = psutil.cpu_percent(1)
        ts  = time.time()
        used_memory = 100 - (psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
        data['cpu'].append(cpu)
        data['ts'].append(ts)
        data['memory'].append(used_memory)
except KeyboardInterrupt:
    df = pd.DataFrame(data=data)
    df.to_csv('monitoring.csv')
