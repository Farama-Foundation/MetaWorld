import random
import time
from functools import partial
from typing import Any, Dict, List, Optional

import gymnasium
import numpy as np

import metaworld
import metaworld.envs

SEED = 42
BENCH_SECONDS = 20

random.seed(SEED)
np.random.seed(SEED)

bench = metaworld.MT50(seed=SEED)


class RandomTaskSelectWrapper(gymnasium.Wrapper):
    """A Gymnasium Wrapper to automatically set / reset the environment to a random
    task."""

    tasks: List[metaworld.Task]
    sample_tasks_on_reset: bool = True

    def _set_random_task(self):
        task_idx = self.np_random.choice(len(self.tasks))
        self.unwrapped.set_task(self.tasks[task_idx])

    def __init__(
        self,
        env: metaworld.SawyerXYZEnv,
        tasks: List[metaworld.Task],
        sample_tasks_on_reset: bool = True,
    ):
        super().__init__(env)
        self.tasks = tasks
        self.sample_tasks_on_reset = sample_tasks_on_reset

    def toggle_sample_tasks_on_reset(self, on: bool):
        self.sample_tasks_on_reset = on

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        if self.sample_tasks_on_reset:
            self._set_random_task()
        return self.env.reset(seed=seed, options=options)

    def sample_tasks(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        self._set_random_task()
        return self.env.reset(seed=seed, options=options)


def make_envs(benchmark: metaworld.Benchmark) -> gymnasium.vector.VectorEnv:
    def _make_env_internal(
        env_cls: type[metaworld.SawyerXYZEnv], env_cls_name: str
    ) -> gymnasium.Env:
        env = env_cls()
        tasks = [
            task for task in benchmark.train_tasks if task.env_name == env_cls_name
        ]
        env = gymnasium.wrappers.TimeLimit(env, env.max_path_length)  # type: ignore
        env = gymnasium.wrappers.RecordEpisodeStatistics(env)  # type: ignore
        env = RandomTaskSelectWrapper(env, tasks)  # type: ignore
        return env

    return gymnasium.vector.AsyncVectorEnv(
        [
            partial(_make_env_internal, env_cls, env_cls_name)
            for env_cls_name, env_cls in benchmark.train_classes.items()
        ]
    )


def main() -> None:
    envs = make_envs(bench)
    print("Made envs!")
    envs.reset()
    print("Envs reset.")

    print("Starting benchmarking...")
    start = time.time()
    current = start
    steps = 0
    while True:
        envs.step(envs.action_space.sample())
        steps += 1
        current = time.time()
        print(
            f"Progress: {(current - start) / BENCH_SECONDS * 100:.2f}%, SPS: {int(steps / (current - start))}",
            end="\r",
        )
        if current - start > BENCH_SECONDS:
            break

    envs.close()
    print("\nFinal SPS", steps // (current - start))


if __name__ == "__main__":
    main()
