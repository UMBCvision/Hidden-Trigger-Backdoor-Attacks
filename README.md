# Hidden-Trigger-Backdoor-Attacks
Official Implementation of the AAAI-20 paper [Hidden Trigger Backdoor Attacks][paper]

With the success of deep learning algorithms in various domains, studying adversarial attacks to secure deep models
in real world applications has become an important research topic. Backdoor attacks are a form of adversarial attacks on
deep networks where the attacker provides poisoned data to the victim to train the model with, and then activates the attack by showing a specific small trigger pattern at the test time. Most state-of-the-art backdoor attacks either provide mislabeled poisoning data that is possible to identify by visual
inspection, reveal the trigger in the poisoned data, or use noise to hide the trigger. We propose a novel form of backdoor attack where poisoned data look natural with correct labels and also more importantly, the attacker hides the trigger in the poisoned data and keeps the trigger secret until the test time. We perform an extensive study on various image classification settings and show that our attack can fool the model by
pasting the trigger at random locations on unseen images although the model performs well on clean data. We also show
that our proposed attack cannot be easily defended using a state-of-the-art defense algorithm for backdoor attacks.

![alt text][teaser]

## Requirements
+ pytorch >=1.3.0

## Dataset creation
```python
python create_imagenet_filelist.py cfg/dataset.cfg
```

+ Change ImageNet data source in dataset.cfg

+ This script partitions the ImageNet train and val data into poison generation, finetune and val to run our backdoor attacks.
Default set to 200 poison generation images, 800 images as finetune and validation images as val.
Change this for your specific needs.

## Configuration file

+ Please create a separate configuration file for each experiment.
+ One example is provided in cfg/experiment.cfg. Create a copy and make desired changes.
+ The configuration file makes it easy to control all parameters (e.g. poison injection rate, epsilon, patch_size, trigger_ID)

## Poison generation
+ First create directory data/<EXPERIMENT_ID> and a file in it named source_wnid_list.txt which will contain all the wnids of the source categories for the experiment. These files are provided for the experiments in the paper. For example data/0011/source_wnid_list.txt corresponds to cfg/singlesource_singletarget_binary_finetune/experiment_0011.cfg. For multi-source attack you can pass a list of multiple source wnids.
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

## Additional results
+ If you want to run the experiments which compare to the BadNets threat model reported in Table 3 of our paper, make a couple of changes
    + In the cfg file, replace poison_root=poison_data with poison_root=patched_data so that the model uses patched data as poisons. The patched data is already saved by generate_poison.py
    + In lines 53-56 of finetune_and_test.py, use a separate checkpoint directory to save the finetuned models so that you don't overwrite models finetuned with our poisons.
    
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

## Acknowledgement
This work was performed under the following financial assistance award: 60NANB18D279 from U.S. Department of Commerce, National Institute of Standards and Technology, funding from SAP SE, and also NSF grant 1845216.

[paper]: https://arxiv.org/abs/1910.00033
[teaser]: https://github.com/UMBCvision/Hidden-Trigger-Backdoor-Attacks/blob/master/Teaser_Updated.png
