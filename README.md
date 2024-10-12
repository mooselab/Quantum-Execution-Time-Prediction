# Quantum-Execution-Time-Prediction
The repository contains the detailed results and replication package for the paper "Understanding and Estimating the Execution Time of Quantum Programs".

In this paper, we first study the characteristics of quantum programs' runtime on simulators and real quantum computers. Then, we introduce an innovative method that employs a graph transformer-based model, utilizing the graph information and global information of quantum programs to estimate their execution time. We selected a benchmark dataset comprising over 1510 quantum programs, initially predicting their execution times on simulators, which yielded promising results with an R-squared value over 95\%. Subsequently, for the estimation of execution times on quantum computers, we applied active learning to select 340 samples with a confidence level of 95\% to build and evaluate our approach, achieving an average R-squared value exceeding 90\%. 


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

## Data Source

The quantum programs dataset used in our project is sourced from the MQTBench, hosted by the Chair of Quantum Technologies at the Technical University of Munich (TUM). More information can be found on their website: [MQTBench](https://www.cda.cit.tum.de/mqtbench/).


## Experiments

To replicate our results, follow the general process outlined below:

1. Execute all quantum programs on simulators to obtain the features and execution times for each program.
2. Use Active Learning to select a subset of quantum programs to execute on real quantum computers, obtaining their features and execution times.
3. Train the model using data obtained from simulators and evaluate the results.
4. Perform cross-validation using data obtained from quantum computers and evaluate the results.
5. Assess the overall performance and results.


## Acknowledgements

Our implimentation bases on or contains many references to following repositories:

* [mqt-bench](https://github.com/cda-tum/mqt-bench)
* [mqt-predictor](https://github.com/cda-tum/mqt-predictor)
* [torchquantum](https://github.com/mit-han-lab/torchquantum)



