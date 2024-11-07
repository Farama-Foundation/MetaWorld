---
layout: "contents"
title: Reward Functions 
firstpage:
---

# Reward Functions

Similar structures are provided with the [action](action_space) and [state space](space_space).
Meta-World provides well-shaped reward functions for the individual tasks that are solveable by current single-task reinforcement learning approaches.
To assure equivalent learning in the settings with multiple tasks, all task rewards have the same magnitude.

## Options

Meta-World currently implements two types of reward functions that can be selected
by passing the `reward_func_version` keyword argument to `gym.make(...)`.

### Version 1

Passing `reward_func_version=v1` configures the benchmark with the primary
reward function of Meta-World, which is actually a version of the
`pick-place-wall` task that is modified to also work for the other tasks.


### Version 2

TBA
