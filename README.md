# VIbCReg

This repository accompanies the VIbCReg paper.
It contains code to reproduce the experiments made in
the paper, and to use the VIbCReg methodology for other
experiments.

### Usage

Before running the experiments, set up a conda 
environment running
```
conda env create -f environment.yaml 
```
Then activate the environment by running
```
conda activate vibcreg 
```
Then download the dataset used in the  experiment
by running
```
python -m vibcreg.data.download_data 
```
Now, you are ready to run experiments. Note that
we use `weights and biases` (https://wandb.ai/) for 
logging the experiments, and you may need to configure
wandb before actually running the experiments.

The default experiment can be run as follows
```
python -m vibcreg.examples.learn_representations 
```
And to  run a specific configuration, run
```
python -m vibcreg.examples.learn_representations --config_dataset config_dataset_filepath --config_framework config_framework_filepath --device_ids "0"
```
After training the model, it is possible to evaluate 
the model by running
```
python -m vibcreg.examples.evaluations
```
with appropriate configuration files, if not using 
the default configs.