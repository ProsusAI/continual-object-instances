
# continual-object-instances
Code for "Continual Learning of Object Instances", Implemented in PyTorch, [https://arxiv.org/abs/2004.10862](https://arxiv.org/abs/2004.10862)


## Abstract
We propose continual instance learning - a method that applies the concept of continual learning to the task of distinguishing instances of the same object category. We specifically focus on the car object, and incrementally learn to distinguish car instances from each other with metric learning. We begin our paper by evaluating current techniques. Establishing that catastrophic forgetting is evident in existing methods, we then propose two remedies. Firstly, we regularise metric learning via Normalised Cross-Entropy. Secondly, we augment existing models with synthetic data transfer. Our extensive experiments on three large-scale datasets, using two different architectures for five different continual learning methods, reveal that Normalised cross-entropy and synthetic transfer leads to less forgetting in existing techniques.


## Authors
Kishan Parshotam and Mert Kilickaya


## Cite
If you make use of this code, please cite our work:
```
@article{parshotam2020continual,
  title={Continual Learning of Object Instances},
  author={Parshotam, Kishan and Kilickaya, Mert},
  journal={arXiv preprint arXiv:2004.10862},
  year={2020}
}
```

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
Cars3D/download.sh
```

Render the 3D dataset with `Cars3D/render.m` script with MATLAB, or another renderer of your choice.

## Experiments
```
usage: main.py [-h] [--data_path DATA_PATH] [-d {Cars3D}] [-ds DATA_SPLITS]
               [-s SAMPLING_METHOD] [-sp SPLIT_METHOD] [-mo {lenet,resnet}]
               [-clm {naive,finetune,lfl,lwf,ewc}]
               [-t {regression,classification}] [-e N_EPOCHS] -o OUTPUT
               [-b BATCH_SIZE] [-l LR] [-lamb LAMBDA_LFL]
               [-lamb_lwf LAMBDA_LWF] [-emb EMBEDDING_DIM] [-im IMAGE_SIZE]
               [-w NUM_WORKERS] [-g GPU] [-p PRINT_EVERY] [-n NEG_SAMPLES]
               [-temp TEMPERATURE] [-lamb_ewc LAMBDA_EWC] [--normalize]
               [--train_full] [--freeze]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        relative path to root data folder
  -d {Cars3D}, --dataset {Cars3D}
                        dataset to process
  -ds DATA_SPLITS, --data_splits DATA_SPLITS
                        number of equal data partitions
  -s SAMPLING_METHOD, --sampling_method SAMPLING_METHOD
                        sampling method
  -sp SPLIT_METHOD, --split_method SPLIT_METHOD
                        split method
  -mo {lenet,resnet}, --model {lenet,resnet}
                        backbone model
  -clm {naive,finetune,lfl,lwf,ewc}, --continuous_learning_method {naive,finetune,lfl,lwf,ewc}
                        continual learning approach
  -t {regression,classification}, --task_method {regression,classification}
                        benchmark or NCE approach
  -e N_EPOCHS, --n_epochs N_EPOCHS
                        define the number of epochs
  -o OUTPUT, --output OUTPUT
                        output folder name folder
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size
  -l LR, --lr LR        learning rate
  -lamb LAMBDA_LFL, --lambda_lfl LAMBDA_LFL
                        lfl weight in euclidean distance between anchors
  -lamb_lwf LAMBDA_LWF, --lambda_lwf LAMBDA_LWF
                        lwf weight in knowledge distillation between anchors
  -emb EMBEDDING_DIM, --embedding_dim EMBEDDING_DIM
                        embedding size
  -im IMAGE_SIZE, --image_size IMAGE_SIZE
                        image size
  -w NUM_WORKERS, --num_workers NUM_WORKERS
                        parallel workers
  -g GPU, --gpu GPU     GPU ID
  -p PRINT_EVERY, --print_every PRINT_EVERY
                        print steps
  -n NEG_SAMPLES, --neg_samples NEG_SAMPLES
                        Number of negative samples for CE loss
  -temp TEMPERATURE, --temperature TEMPERATURE
                        Temperature for softmax under the NCE setting
  -lamb_ewc LAMBDA_EWC, --lambda_ewc LAMBDA_EWC
                        lambda ewc
  --normalize           normalize network outputs
  --train_full          cumulative/offline training
  --freeze              freeze conv layers, used in Naive approach
```



### Benchmarking experiment

#### Offline training
```bash
python src/main.py -o OUTPUT --data_path DATA_PATH -d Cars3D -ds 1 -m MODEL -t regression
```

#### Continual training
```bash
python src/main.py -o OUTPUT --data_path DATA_PATH -d Cars3D -ds 10 -clm CONTINUOUS_LEARNING_METHOD -m MODEL -t regression
```


### Normalized Cross Entropy
```bash
python src/main.py -o OUTPUT --data_path DATA_PATH -d Cars3D -ds 10 -clm CONTINUOUS_LEARNING_METHOD -m MODEL -t classification 
```


## Disclaimer
This is not an official Prosus product. It is the outcome of an internal research project from the Prosus AI team.


### About Prosus 
Prosus is a global consumer internet group and one of the largest technology investors in the world. Operating and
 investing globally in markets with long-term growth potential, Prosus builds leading consumer internet companies that empower people and enrich communities.
For more information, please visit [www.prosus.com](www.prosus.com).

## Contact information
Please contact Kishan Parshotam `kishanarendra[at]gmail[dot]com` for issues and questions.