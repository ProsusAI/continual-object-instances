
# continual-object-instances

Code for "Continual Learning of Object Instances", Implemented in PyTorch, [https://arxiv.org/abs/2004.10862](https://arxiv.org/abs/2004.10862)

## Abstract

We propose continual instance learning - a method that applies the concept of continual learning to the task of distinguishing instances of the same object category. We specifically focus on the car object, and incrementally learn to distinguish car instances from each other with metric learning. We begin our paper by evaluating current techniques. Establishing that catastrophic forgetting is evident in existing methods, we then propose two remedies. Firstly, we regularise metric learning via Normalised Cross-Entropy. Secondly, we augment existing models with synthetic data transfer. Our extensive experiments on three large-scale datasets, using two different architectures for five different continual learning methods, reveal that Normalised cross-entropy and synthetic transfer leads to less forgetting in existing techniques.


## Authors

Kishan Parshotam and Mert Kilickaya


## Cite

................................


## Installing
Install the dependencies by creating the Conda environment `continual_objects` from the given `environment.yml` file and activating it.
```bash
conda env create -f environment.yml
conda activate continual_objects
```

## Datasets
Downloading the Cars3D dataset:
```bash
chmod +x Cars3D/download.sh
./download.sh
```

Render the 3D dataset with `Cars3D/render.m` script with MATLAB, or another renderer of your choice.

## Experiments


### Benchmarking


### Normalized Cross Entropy


### Synthetic transfer model


## Disclaimer
This is not an official Prosus product. It is the outcome of an internal research project from the Prosus AI team.

### About Prosus 
Prosus is a global consumer internet group and one of the largest technology investors in the world. Operating and
 investing globally in markets with long-term growth potential, Prosus builds leading consumer internet companies that empower people and enrich communities.
For more information, please visit [www.prosus.com](www.prosus.com).

## Contact information
Please contact Kishan Parshotam `kishanarendra[at]gmail[dot]com` for issues and questions.