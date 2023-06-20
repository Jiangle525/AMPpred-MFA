**Directory Structure**

```
├── 2. Data set information.ipynb
├── 3. Attention visualization.ipynb
├── 3. Feature visualization.ipynb
├── 3. Ablation experiment.ipynb
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

- `2. Data set information.ipynb`: Analyze dataset information.

- `3. Attention visualization.ipynb`: `UniProt entry` is the attention visualization of `A0A1P8AQ95`, including generating attention matrix heatmap, attention feature sorting, attention network, etc.

- `3. Feature visualization.ipynb`: Analyze the feature extraction process of the first 3000 samples in the training set.

- `3. Ablation experiment.ipynb`: Attention ablation experiment and k-mer experiment.