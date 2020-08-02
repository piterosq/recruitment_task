#! /bin/bash
sudo apt-get install yum subscription-manager dnf rpm -y
#installing the NVIDIA driver
yum install https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm

subscription-manager repos --enable=rhel-8-for-x86_64-appstream-rpms
subscription-manager repos --enable=rhel-8-for-x86_64-baseos-rpms
subscription-manager repos --enable=rhel-8-for-x86_64-crb-rpms

#action is needed here
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-rhel8-11-0-local-11.0.2_450.51.05-1.x86_64.rpm

sudo rpm -i cuda-repo-rhel8-11-0-local-11.0.2_450.51.05-1.x86_64.rpm

sudo yum clean expire-cache

sudo dnf clean expire-cache
sudo dnf module install nvidia-driver:latest-dkms
sudo dnf install cuda

export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

sudo yum install -y nvidia-container-toolkit
sudo systemctl restart docker

docker pull nvcr.io/nvidia/pytorch:20.07-py3

docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/pytorch:20.07-py3
