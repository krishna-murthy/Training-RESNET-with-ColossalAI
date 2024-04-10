# Training-RESNET-with-ColossalAI

This repository provides scripts for training the ResNet-18 model on the CIFAR10 dataset using ColossalAI. Included are experiment logs demonstrating the execution of this example within a Google Colab notebook environment, utilizing `nproc_per_node=1` over a span of 10 epochs.

## Installation Requirements

Before proceeding, ensure that you have all necessary dependencies installed. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Training the model

To train the model, we utilize PyTorch's Distributed Data Parallel (DDP) with floating point 32 (FP32) precision. 

*Please note that the `nproc_per_node` value should be adjusted according to the resources available in your environment*

Execute the following command to start the training process:
```bash
colossalai run --nproc_per_node 1 train_resnet_cifar10.py
```
## Expected accuracy

After training for 10 epochs, the expected accuracy performance for the ResNet-18 model is as follows:

| Model	   | Single-GPU Baseline FP32 |
|----------|--------------------------|
|ResNet-18 |	75.07%                  |

