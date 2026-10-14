import numpy as np
from pfn_network.env.core import ParallelEnvs
from pfn_network.env.dgp import TaskPrior, sample_task

def test_spawn_workers_match_serial_environments():
    prior = TaskPrior(n=5, budget=2, catalog_size=6, horizon=2)
    bundles = [sample_task(prior, 7, 'train', i, d) for i, d in enumerate(['FC', 'CD'])]
    serial = [bundle.env(i) for i, bundle in enumerate(bundles)]
    tasks = [(bundle.env(i), [i]) for i, bundle in enumerate(bundles)]
    with ParallelEnvs(2) as workers:
        for _ in range(2):
            workers.load(tasks)
            observations = workers.reset([0, 1])
            expected = [env.reset(seed=i)[0] for i, env in enumerate(serial)]
            for got, want in zip(observations, expected):
                for key in ['pfn_state', 'global_state', 'valid_action_mask', 'catalog']:
                    np.testing.assert_array_equal(got[key], want[key])
            for _ in range(prior.horizon):
                results = workers.step([(0, 0), (1, 0)])
                for i, env in enumerate(serial):
                    want = env.step(0)
                    got = results[i]
                    assert got[1:] == want[1:]
                    np.testing.assert_array_equal(got[0]['pfn_state'], want[0]['pfn_state'])
