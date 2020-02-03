#!/bin/bash

cd ../..

CUDA_VISIBLE_DEVICES=1 python generate_poison.py cfg/singlesource_singletarget_1000class_finetune/experiment_0002.cfg
CUDA_VISIBLE_DEVICES=1 python finetune_and_test.py cfg/singlesource_singletarget_1000class_finetune/experiment_0002.cfg