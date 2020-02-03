#!/bin/bash

cd ..
cd ..

CUDA_VISIBLE_DEVICES=1 python generate_poison.py cfg/singlesource_singletarget_binary_finetune/experiment_0012.cfg
CUDA_VISIBLE_DEVICES=1 python finetune_and_test.py cfg/singlesource_singletarget_binary_finetune/experiment_0012.cfg