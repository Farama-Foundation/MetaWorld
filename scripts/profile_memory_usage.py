#!/usr/bin/env python3
"""Test script for profiling average memory footprint."""
import memory_profiler

from metaworld.envs.env_dict import ALL_V3_ENVIRONMENTS
from tests.helpers import step_env


def build_and_step(env_cls):
    env = env_cls()
    env.reset()
    step_env(env, max_path_length=1000, iterations=10)
    return env


def build_and_step_all(classes):
    envs = []
    for env_cls in classes:
        env = build_and_step(env_cls)
        envs += [env]


def profile_hard_mode_indepedent():
    profile = {}
    for env_cls in ALL_V3_ENVIRONMENTS:
        target = (build_and_step, [env_cls], {})
        memory_usage = memory_profiler.memory_usage(target)
        profile[env_cls] = max(memory_usage)

    return profile


def profile_hard_mode_shared():
    target = (build_and_step_all, [ALL_V3_ENVIRONMENTS], {})
    usage = memory_profiler.memory_usage(target)
    return max(usage)


if __name__ == "__main__":
    profile = profile_hard_mode_indepedent()
    print("--------- Independent memory footprints ---------")
    for cls, u in profile.items():
        print(f"{cls.__name__:<40} {u:>5.1f} MB")
    max_independent = max(profile.values())
    mean_independent = sum(profile.values()) / len(profile)
    min_independent = min(profile.values())
    print("\nSummary:")
    print("| min      | mean     | max      |")
    print("|----------|----------|----------|")
    print(
        f"| {min_independent:.1f} MB | {mean_independent:.1f} MB | {max_independent:.1f} MB |"
    )
    print("\n")

    print("---------    Shared memory footprint    ---------")
    max_usage = profile_hard_mode_shared()
    mean_shared = max_usage / len(ALL_V3_ENVIRONMENTS)
    print(
        f"Mean memory footprint (n = {len(ALL_V3_ENVIRONMENTS)}): {mean_shared:.1f} MB"
    )
