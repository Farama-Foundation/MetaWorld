---
layout: "contents"
title: Basic Usage
firstpage:
---

# Basic Usage

## Using the benchmark
There are 6 major benchmarks pre-packaged into Meta-World with support for making your own custom benchmarks. The benchmarks are divided into Multi-Task and Meta reinforcement learning benchmarks.

### Multi-Task Benchmarks
The MT1, MT10, and MT50 benchmarks are the Multi-Task Benchmarks. These benchmarks are used to learn a multi-task policy that can learn 1, 10, or 50 training tasks simultaneously. MT1 benchmarks can be created with any of the 50 tasks available in Meta-World.
In the MT10 and MT50 benchmarks, the observations returned by the benchmark will come with one-hot task IDs appended to the state.

### Meta-Learning Benchmarks
The ML1, ML10, and ML45 benchmarks are 3 meta-reinforcement learning benchmarks available in Meta-World. The ML1 benchmark can be used with any of the 50 tasks available in Meta-World.
The ML1 benchmark tests for few-shot adaptation to goal variations within a single task. The ML10 and ML45 both test few-shot adaptation to new tasks. ML10 comprises 10 train tasks with 5 test tasks, while ML45 comprises of 45 training tasks with 5 test tasks.


### MT1
```python
import gymnasium as gym
import metaworld

SEED = 0  # some seed number here
env = gym.make('Meta-World/MT1-reach', seed=seed)
obs, info = env.reset()

a = env.action_space.sample() # randomly sample an action
obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action
```

### MT10
MT10 has two different versions that can be returned by gym.make. The first version is the synchronous version of the benchmark where all environments are contained within the same process.
For users with limited compute resources, the synchronous option needs the least resources.
```python
import gymnasium as gym
import metaworld

seed = 42

envs = gym.make('Meta-World/MT10-sync', seed=seed) # this returns a Synchronous Vector Environment with 10 environments

obs, info = envs.reset() # reset all 10 environments

a = env.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 10 environments
```
Alternatively, for users with more compute we also provide the asynchronous version of the MT10 benchmark where each environment is isolated in it's own process and must use inter-process messaging via pipes to communicate.

```python
envs = gym.make('Meta-World/MT10-async', seed=seed) # this returns an Asynchronous Vector Environment with 10 environments
```

### MT50
MT50 also contains two different versions, a synchronous and an asynchronous version, of the environments.
```python
import gymnasium as gym
import metaworld

seed = 42

envs = gym.make('Meta-World/MT50-sync', seed=seed) # this returns a Synchronous Vector Environment with 50 environments

obs, info = envs.reset() # reset all 50 environments

a = env.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 50 environments
```

```python
envs = gym.make('Meta-World/MT50-async', seed=seed) # this returns an Asynchronous Vector Environment with 50 environments
```


## Meta-Learning Benchmarks
Each Meta-reinforcement learning benchmark has training and testing environments. These environments must be created separately as follows.

### ML1
```python
import gymnasium as gym
import metaworld

seed = 42

train_envs = gym.make('Meta-World/ML1-train-reach-V3', seed=seed)
test_envs = gym.make('Meta-World/ML1-test-reach-V3', seed=seed)

# training procedure use train_envs
# testing procedure use test_envs

```


### ML10
Similar to the Multi-Task benchmarks, the ML10 and ML45 environments can be run in synchronous or asynchronous modes.


```python
import gymnasium as gym
import metaworld
train_envs = gym.make('Meta-World/ML10-train-sync', seed=seed) # or ML10-train-async
test_envs = gym.make('Meta-World/ML10-test-sync', seed=seed) # or ML10-test-async
```


### ML45
```python
import gymnasium as gym
import metaworld

train_envs = gym.make('Meta-World/ML45-train-sync', seed=seed) # or ML45-train-async
test_envs = gym.make('Meta-World/ML45-test-sync', seed=seed) # or ML45-test-async
```


## Custom Benchmarks
Finally, we also provide support for creating custom benchmarks by combining any number of Meta-World environments.

The prefix 'mt' will return environments that are goal observable for Multi-Task reinforcement learning, while the prefix 'ml' will return environments that are partially observable for Meta-reinforcement learning.
Like the included MT and ML benchmarks, these environments can also be run in synchronous or asynchronous mode.
In order to create a custom benchmark, the user must provide a list of environment names with the suffix '-V3'.

```python
import gymnasium as gym
import metaworld

envs = gym.make('Meta-World/mt-custom-sync', envs_list=['env_name_1-V3', 'env_name_2-V3', 'env_name_3-V3'], seed=seed)
```

## Arguments
