#!/bin/bash

cd ..
cd ..

# CUDA_VISIBLE_DEVICES=2 python generate_poison.py cfg/singlesource_singletarget_binary_finetune/experiment_0013.cfg
CUDA_VISIBLE_DEVICES=2 python finetune_and_test_BCEloss_figures.py cfg/singlesource_singletarget_binary_finetune/experiment_0013.cfg