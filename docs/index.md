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
import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('pick-place-V3') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-V3']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, terminate, truncate, info = env.step(a)
```

```{toctree}
:hidden:
:caption: Introduction

introduction/basic_usage
evaluation/evaluation
installation/installation
rendering/rendering
usage/basic_usage
```


```{toctree}
:hidden:
:caption: Development

Github <https://github.com/Farama-Foundation/Metaworld>
citation
release_notes/index
Contribute to the Docs <https://github.com/Farama-Foundation/Metaworld/blob/main/docs/README.md>
```
