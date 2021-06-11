from main import *


tests = []
results = []
tests.append(TestRunSettings(title='RR centralized', placement_policy='central', lb_type='rr', duration=300, req_per_sec=50))
tests.append(TestRunSettings(title='RR distributed', placement_policy='dist', lb_type='rr', duration=300, req_per_sec=50))
tests.append(TestRunSettings(title='LRT centralized', placement_policy='central', lb_type='lrt', duration=300, req_per_sec=50))
tests.append(TestRunSettings(title='LRT distributed', placement_policy='dist', lb_type='lrt', duration=300, req_per_sec=50))

for t in tests:
    dfs = make_test_run(t)
    results.append((t.title, dfs))

for title, dfs in results:
    make_analysis(dfs, title)
