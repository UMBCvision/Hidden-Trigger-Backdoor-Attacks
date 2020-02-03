#!/bin/bash

cd ..
cd ..

CUDA_VISIBLE_DEVICES=3 python generate_poison.py cfg/singlesource_singletarget_binary_finetune/experiment_0014.cfg
CUDA_VISIBLE_DEVICES=3 python finetune_and_test.py cfg/singlesource_singletarget_binary_finetune/experiment_0014.cfg