# Prodigal: Backdoor Defense for Federated Learning beyond Robust Aggregation

Code for the following paper:
"Prodigal: Backdoor Defense for Federated Learning beyond Robust Aggregation".

## Introduction

Federated learning is susceptible to backdoor attacks due to its stealthiness. Existing robust aggregation defenses, e.g., RLR and RFA, utilize robust statistics to filter out backdoored parameters/gradients in each aggregation step. However, as evidenced by many recent literature, these robust aggregation solutions are too fragile to defend against backdoor attacks. First time in the literature, we discover that their fundamental reason for failure is due to the immediate removal of the poisoned parameters in each aggregation step, because this operation will trigger the malicious clients to transfer the poisoning effect to other coordinates, which we name poisoning transfer effect. With this important finding, we propose a robust federated backdoor defense that breaks the convention of robust aggregation by keeping poisoning parameters in each aggregation step. Prodigal follows two key steps: First, it uses the TOPK gradient as a robust statistic to accurately identify poisoned parameters. Second, instead of removing those poisoned parameters in each aggregation phase, the poisoned parameters are kept in the model and distributed to the attackers, which prevents the poisoning from transferring to other benign parameters. Experiments show that our defense achieves much more robust defense performance compared to existing robust aggregation solutions.
## Requirements
* Python 3.8
* PyTorch 1.13

## Usage

### Train ResNet9 on CIFAR10/CIFATR100/TinyImageNet:
Running ``fl_experiments/standalone/Isolation/isolation_exp.py``.


### Parameters

All training/inference/model parameters are controlled from ``fl_experiments/standalone/Isolation/isolation_exp.py``.
