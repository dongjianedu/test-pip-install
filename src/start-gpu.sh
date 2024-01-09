#!/bin/bash




echo "test gpu 2.1.0"

echo "---------------env------------------"
env
echo "---------------env------------------"


echo "---------------libnvidia-ml------------------"
ls /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1
echo "---------------libnvidia-ml------------------"

cd /usr/lib/x86_64-linux-gnu/
libnvidia=$(find . -name "libnvidia-ml.so*" | grep -v "libnvidia-ml.so.1")
version=${libnvidia#*libnvidia-ml.so.}
echo $version

cp libcuda.so libcuda.so.backup
rm libcuda.so
ln -s libcuda.so.1 libcuda.so

# 建立软链接 libcuda.so.1 > libcuda.so.450.80.02
cp libcuda.so.1 libcuda.so.1.backup
rm libcuda.so.1
cp libcuda.so.$version libcuda.so.1

# 建立软链接 libnvidia-ml.so.1 > libnvidia-ml.so.450.80.02
cp libnvidia-ml.so.1 libnvidia-ml.so.1.backup
rm libnvidia-ml.so.1
ln -s libnvidia-ml.so.$version libnvidia-ml.so.1



nvidia-smi


