#!/bin/bash

echo "test gpu 2.1.0"

echo "---------------env------------------"
env
echo "---------------env------------------"


echo "---------------libnvidia-ml------------------"
ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
echo "---------------libnvidia-ml------------------"

nvidia-smi
