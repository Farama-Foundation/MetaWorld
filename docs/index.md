---
hide-toc: true
firstpage:
lastpage:
---

```{project-logo} _static/metaworld-text.svg
:alt: Metaworld Logo
```

```{project-heading}
Meta-World is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distinct robotic manipulation tasks.
```

```{figure} _static/mt10.gif
   :alt: REPLACE ME
   :width: 500
```

**Basic example:**

```{code-block} python
import gymnasium as gym
import metaworld

env = gym.make('MetaWorld/reach-v3')

obs = env.reset()
a = env.action_space.sample()
next_obs, reward, terminate, truncate, info = env.step(a)

```

```{toctree}
:hidden:
:caption: Introduction

introduction/basic_usage
evaluation/evaluation
installation/installation
rendering/rendering
```

```{toctree}
:hidden:
:caption: Benchmark Information
benchmark/environment_creation
benchmark/action_space
benchmark/state_space
benchmark/benchmark_descriptions
benchmark/task_descriptions.md
benchmark/reward_functions
benchmark/expert_trajectories
benchmark/resetting
```

```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Metaworld>
citation
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/Metaworld/blob/main/docs/README.md>
```
