# self-adaptive-data-shifting
kNN based self-adaptive data shifting for one-class support vector (OCSVM) machine hyperparameter selection

This code is written based on the methods described in [1]. The methods include two parts, "Negative shifting" to generate pseudo outliers, and "Positive shifting" to generate pseudo target data.

In the code, a "banana" training dataset from https://www.openml.org/d/1460 is used.

## Requirements
* Python 2 or 3
* numpy>=1.13
* scikit_learn>=0.19.1
* matplotlib 2.2.x or 3.0

## References
[1] Wang, S., Liu, Q., Zhu, E., Porikli, F., & Yin, J. (2018). Hyperparameter selection of one-class support vector machine by self-adaptive data shifting. Pattern Recognition, 74, 198-211.
