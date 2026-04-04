import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ARTIFACTS_BASE_DIR = '../notebooks/artifacts'
MODELS_DIR = f'{ARTIFACTS_BASE_DIR}/models/rl_tasks'
RESULTS_DIR = f'{ARTIFACTS_BASE_DIR}/results/agent_evaluations/rl_tasks'
LOGS_DIR = f'{ARTIFACTS_BASE_DIR}/logs/rl_tasks'

SEEDS = [0, 8, 16, 24, 32]

TASK_SCRIPTS = [
    'auto_got_3_1_ppo_sum_list.py',
    'auto_got_3_2_ppo_sort_list.py',
    'auto_got_3_3_ppo_count_keywords.py',
    'auto_got_3_4_ppo_intersect_set.py',
    'auto_got_3_5_ppo_merge_docs.py',
]


def run_task_seed(script_name: str, seed: int) -> tuple[str, int, int, float]:
    log_path = f'{LOGS_DIR}/{script_name}_s{seed}.log'
    script_path = f'./{script_name}'

    print(f'[{script_name} s{seed}] Starting - log: {log_path}')
    start = time.time()

    with open(log_path, 'w') as log_file:
        proc = subprocess.run(
            [sys.executable, script_path, '--seed', str(seed)],
            cwd='.',
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - start
    status = 'OK' if proc.returncode == 0 else f'FAILED (exit {proc.returncode})'
    print(f'[{script_name} s{seed}] {status} — {elapsed:.1f}s')
    return script_name, seed, proc.returncode, elapsed


def _ensure_dirs() -> None:
    for directory in [
        ARTIFACTS_BASE_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        MODELS_DIR,
        LOGS_DIR
    ]:
        os.makedirs(directory, exist_ok=True)


def main() -> None:
    _ensure_dirs()
    jobs = [(script, seed) for script in TASK_SCRIPTS for seed in SEEDS]

    max_workers = min(len(jobs), os.cpu_count() or 1)

    print(f'Running {len(jobs)} jobs ({len(TASK_SCRIPTS)} tasks x {len(SEEDS)} seeds) '
          f'with up to {max_workers} parallel workers.')

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_task_seed, script, seed): (script, seed) for script, seed in jobs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                script, seed = futures[future]
                print(f'[{script} s{seed}] Unexpected error: {exc}')
                results.append((script, seed, -1, 0.0))

    for task_stem, seed, returncode, elapsed in sorted(results):
        status = 'OK' if returncode == 0 else f'FAILED ({returncode})'
        print(f'  {task_stem:<45} s{seed}  {status}  {elapsed:.1f}s')


if __name__ == '__main__':
    main()
