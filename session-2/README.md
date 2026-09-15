# Session 2

Train your own model with a custom dataset ([Chinese MNIST](https://www.kaggle.com/gpreda/chinese-mnist), [direct download](https://www.kaggle.com/api/v1/datasets/download/gpreda/chinese-mnist)), using hyperparameter tuning.


## Installation
### With Conda
Create a conda environment by running

```bash
conda create --name aidl-session2 python=3.11
```
Then, activate the environment
```bash
conda activate aidl-session2
```
and install the dependencies
```bash
pip install -r requirements.txt
```

**Note:** it is important to create a new conda environment, and not re-use previous ones from other sessions.

## Running the project

To run the project, run
```bash
python session-2/main.py
```
To run the project with hyperparameter tuning, run
```bash
python session-2/main_hyperparam_optimize.py
```
