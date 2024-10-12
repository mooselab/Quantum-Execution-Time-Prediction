# Quantum-Execution-Time-Prediction
The repository contains the detailed results and replication package for the paper "Understanding and Estimating the Execution Time of Quantum Programs".


## Repository Structure
This repository is organized into the following folders:

- **data_preparation/**: Contains the code for data preprocessing.
- **model/**: Contains the codes used to train the graph transformer model.
- **data/**: Contains both the raw and processed data.
- **results/**: Contains the saved models and experimental results.


## Dependencies
We recommend using an Anaconda environment with Python version 3.9, and following Python requirement should be met.

* Numpy 1.23.5
* Pandas 2.1.4
* PyTorch 1.13.1
* Qiskit 0.44.2
* Torch_geometric 2.5.2
* Sklearn 1.3.0

## Dataset

### Source
We use HDFS, BGL, Spirit and Thunderbird datasets. 
Original datasets are accessed from [LogHub](https://github.com/logpai/loghub) project.
(We do not provide generated log representations as they are in huge size. Please generate them with our codes provided.)

Due to computational limitations, we utilized subsets of the Spirit and Thunderbird datasets in our experiments. These subsets are available for access at [Zenodo](https://doi.org/10.5281/zenodo.7851024).

### Extra regular expression parsed to the Drain parser

We used Drain to parse the studied datasets. We adopted the default parameters from the following paper for parsing.

```
Pinjia He, Jieming Zhu, Zibin Zheng, and Michael R. Lyu. Drain: An Online Log Parsing Approach with Fixed Depth Tree, Proceedings of the 24th International Conference on Web Services (ICWS), 2017.
```

However, Drain parser generated too much templates with the default setting due to the failure of spotting some dynamic fields. We passed the following regular expression to reduce the amount.

For BGL dataset:

For configuration used in our experiment:
```
regex      = [r'core\.\d+',
              r'(?<=r)\d{1,2}',
              r'(?<=fpr)\d{1,2}',
              r'(0x)?[0-9a-fA-F]{8}',
              r'(?<=\.\.)0[xX][0-9a-fA-F]+',
              r'(?<=\.\.)\d+(?!x)',
              r'\d+(?=:)',
              r'^\d+$',  #only numbers
              r'(?<=\=)\d+(?!x)',
              r'(?<=\=)0[xX][0-9a-fA-F]+',
              r'(?<=\ )[A-Z][\+|\-](?= |$)',
              r'(?<=:\ )[A-Z](?= |$)',
              r'(?<=\ [A-Z]\ )[A-Z](?= |$)'
              ]
```

We refined the RegExps for more accurate parsing as follows:

```
              r'core\.\d+',
              r'(?<=:)(\ [A-Z][+-]?)+(?![a-z])', # match X+ A C Y+......
              r'(?<=r)\d{1,2}',
              r'(?<=fpr)\d{1,2}',
              r'(0x)?[0-9a-fA-F]{8}',
              r'(?<=\.\.)0[xX][0-9a-fA-F]+',
              r'(?<=\.\.)\d+(?!x)',
              r'\d+(?=:)',
              r'^\d+$',  #only numbers
              r'(?<=\=)\d+(?!x)',
              r'(?<=\=)0[xX][0-9a-fA-F]+'  # for hexadecimal
```

For Spirit dataset:
```
regex      = [r'^\d+$',  #only numbers
              r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}[^0-9]',   # IP address
              r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',   # MAC address
              r'\d{14}(.)[0-9A-Z]{10,}',   # message id
              r'(?<=@#)(?<=#)\d+',   #  message id special format
              r'[0-9A-Z]{10,}', # id
              r'(?<=:|=)(\d|\w+)(?=>|,| |$|\\)'   # parameter after:|=
             ]
```

For Thunderbird dataset:

```
regex      = [
             r'(\d+\.){3}\d+',
             r'((a|b|c|d)n(\d){2,}\ ?)+', # a|b|c|dn+number
             r'\d{14}(.)[0-9A-Z]{10,}@tbird-#\d+#', # message id
             r'(?![0-9]+\W)(?![a-zA-Z]+\W)(?<!_|\w)[0-9A-Za-z]{8,}(?!_)',      # char+numbers,
             r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # ip address
             r'\d{8,}',   # numbers + 8
             r'(?<=:)(\d+)(?= )',    # parameter after :
             r'(?<=pid=)(\d+)(?= )',   # pid=XXXXX
             r'(?<=Lustre: )(\d+)(?=:)', # Lustre:
             r'(?<=,)(\d+)(?=\))'
             ]
```

## Experiments

The general process to replicate our results is:

1. Generate structured parsed dataset using [loglizer](https://github.com/logpai/loglizer) with Drain parser into JSON format.
2. Split the dataset into training and testing set and save as NPZ format, with `x_train`, `y_train`, `x_test`, `y_test`.
3. Generate selected log representations with corresponding codes within the `logrep` folder, and generates representations and save as NPY or NPZ format.
4. If the studied technique generates event-level representations, use the `aggregation.py` in the `logrep` folder to merge them into sequence-level for the models that demand sequence-level input.
5. Load generated representations and corresponding labels, and run the models within the `models` folder to get the results.

* Sample parsed data and splitted data are provided in `samples` folder.


## Network details for CNN and LSTM

### CNN
|    Layer    |                          Parameters                          |      Output      |
| :---------: | :----------------------------------------------------------: | :--------------: |
|  __Input__  |                 _win\_size * Embeddin\_size_                 |       N/A        |
|   __FC__    |                    _Embedding\_size * 50_                    | _Win\_size * 50_ |
| __Conv 1__  | _kernel_size=[3, 50], stride=[1, 1], padding=valid, MaxPool2D:[𝑤𝑖𝑛_𝑠𝑖𝑧𝑒 − 3, 1], LeakyReLU_ |   _50 * 1 * 1_   |
| __Conv 2__  | _kernel\_size=[4, 50], stride=[1, 1], padding=valid, MaxPool2D: [𝑤𝑖𝑛_𝑠𝑖𝑧𝑒 − 3, 1], LeakyReLU_ |   _50 * 1 * 1_   |
| __Conv 3__  | _kernel\_size=[5, 50], stride=[1, 1], padding=valid, MaxPool2D:[𝑤𝑖𝑛_𝑠𝑖𝑧𝑒 − 4, 1], LeakyReLU_ |   _50 * 1 * 1_   |
| __Concat__  | _Concatenate feature maps of Conv1, Conv2, Conv3, Dropout(0.5)_ |  _150 * 1 * 1_   |
|   __FC__    |                         _[150 * 2]_                          |       $2$        |
| __Output__ |                           _Softmax_                           |                  |



### LSTM

|   Layer    |           Parameters            |        Output         |
| :--------: | :-----------------------------: | :-------------------: |
| __Input__  | _[win\_size * Embedding\_size]_ |         _N/A_         |
|  __LSTM__  |        _Hidden\_dim = 8_        | _Embedding\_size * 8_ |
|   __FC__   |            _[8 * 2]_            |          _2_          |
| __Output__ |            _Softmax_            |                       |


## Acknowledgements

Our implimentation bases on or contains many references to following repositories:

* [logparser](https://github.com/logpai/logparser)
* [loglizer](https://github.com/logpai/loglizer)
* [deep-loglizer](https://github.com/logpai/deep-loglizer)
* [ScottKnottESD](https://github.com/klainfo/ScottKnottESD)
* [clip(bert)-as-service](https://github.com/jina-ai/clip-as-service)
* [imbalanced-dataset-sampler](https://github.com/ufoym/imbalanced-dataset-sampler)

## Citing & Contacts

Please cite our work if you find it helpful to your research.

Wu, X., Li, H. & Khomh, F. On the effectiveness of log representation for log-based anomaly detection. Empir Software Eng 28, 137 (2023). https://doi.org/10.1007/s10664-023-10364-1

```
@article{article,
year = {2023},
month = {10},
pages = {},
title = {On the effectiveness of log representation for log-based anomaly detection},
volume = {28},
journal = {Empirical Software Engineering},
doi = {10.1007/s10664-023-10364-1}
}
```


