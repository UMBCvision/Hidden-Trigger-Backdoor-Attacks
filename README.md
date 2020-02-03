# Hidden-Trigger-Backdoor-Attacks
Official Implementation of the AAAI-20 paper "Hidden Trigger Backdoor Attacks"

## Requirements
+ pytorch >=1.3.0

## Dataset creation
```python
python create_imagenet_filelist.py cfg/dataset.cfg
```

+ Change ImageNet data source in dataset.cfg

+ This script partitions the ImageNet train and val data into poison generation, finetune and val to run our backdoor attacks.
Default set to 200 poison generation images, remaining images from train as finetune and validation images as val.
Change this for your specific needs.

## Configuration file

+ Please create a separate configuration file for each experiment.
+ One example is provided in cfg/experiment.cfg. Create a copy and make desired changes.
+ The configuration file makes it easy to control all parameters (e.g. poison injection rate, epsilon, patch_size, trigger_ID)

## Poison generation
```python
python generate_poison.py cfg/experiment.cfg
```

## Finetune and test
```python
python finetune_and_test.py cfg/experiment.cfg
```

## Data

+ We have provided the triggers used in our experiments in data/triggers
+ To reproduce our experiments use configuration files given in cfg/singlesource_singletarget_binary_finetune. Please use the correct poison injection rates. There might be some variation in numbers depending on the randomness of the ImageNet data split.

## Shell scripts
+ We also provide shell scripts for ease of experiments. They can be found in scripts/singlesource_singletarget_binary_finetune.

## Citation
If you find our paper or code useful, please cite us using
```bib
@article{saha2019hidden,
  title={Hidden Trigger Backdoor Attacks},
  author={Saha, Aniruddha and Subramanya, Akshayvarun and Pirsiavash, Hamed},
  journal={arXiv preprint arXiv:1910.00033},
  year={2019}
}
```
