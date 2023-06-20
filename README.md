#### Directory Structure

```
├── Data set information.ipynb
├── Attention visualization.ipynb
├── Feature visualization.ipynb
├── Ablation experiment.ipynb
├── AMPpred_MFA
├── dataset
├── README.md
├── multiple_training.py
├── trained_model
└── training_and_testing.py
```

- `AMPpred_MFA`: Python package, storing `AMPpred-MFA` related codes

- `dataset`: All datasets for experiments

- `multiple_training.py`: Specify positive and negative samples, construct training set and test set for multiple training. Use the following command to see how to use it.

  ```shell
  python multiple_training.py --help
  ```

- `training_and_testing.py`: Specify training set and test set, single training. Use the following command to see how to use it.

  ```shell
  python training_and_testing.py --help
  ```

- `Data set information.ipynb`: Analyze dataset information.

- `Attention visualization.ipynb`: `UniProt entry` is the attention visualization of `A0A1P8AQ95`, including generating attention matrix heatmap, attention feature sorting, attention network, etc.

- `Feature visualization.ipynb`: Analyze the feature extraction process of the first 3000 samples in the training set.

- `Ablation experiment.ipynb`: Attention ablation experiment and k-mer experiment.

#### Web server

Free online prediction can be accessed via http://auligey.vip:2027/

