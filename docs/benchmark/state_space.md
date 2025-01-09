---
layout: "contents"
title: State Space 
firstpage:
---

# State Space


Likewise the [action space](action_space), the state space among the tasks requires maintaining the same structure that allows current approaches to employ a single policy/model.
Meta-World contains tasks that either require manipulation of a single object with a potentially variable goal position (e.g., `reach`, `push`, `pick place`) or two objects with a fixed goal position (e.g., `hammer`, `soccer`, `shelf place`).
To account for such variability, large parts of the observation space are kept as placeholders, e.g., for the second object, if only one object is available.

The observation array consists of the end-effector's 3D Cartesian position and the composition of a single object with its goal coordinates or the positions of the first and second object.
This always results in a 9D state vector.

TODO: Provide table
