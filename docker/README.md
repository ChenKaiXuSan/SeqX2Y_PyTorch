# How to deploy the environment to your local machine

## pull the image from docker hub
first you need to install docker, desktop version or server version, it's up to you.

pull the image from docker hub, here we use the latest version of the pytorch image.

``` bash
docker pull pytorch/pytorch:latest
```

<span id="run">then run the image, here we use the interactive mode, and mount the current directory to the container.</span>

``` bash
docker run -itd -v $(pwd)/{your_fold_path}:/workspace --gpus all --name {your_container_name} pytorch/pytorch:latest bash
```

then you can use the following command to enter the container.

``` bash
docker exec -it {container_id} bash
```

when enter the container, you can use the following command to install the requirements.

``` bash
cd /workspace
pip install -r requirements.txt
```

then you can run the code in the container.

``` bash
python project/main.py
```

next have a cup of coffee and wait for the result.
have a nice day!

## build the image from the dockerfile
here we also support you to build the image from the dockerfile.

1. git clone the project to your local machine.

``` bash
git clone https://github.com/ChenKaiXuSan/SeqX2Y_PyTorch.git
```
2. change the directory to the docker folder.

``` bash
cd  SeqX2Y_PyTorch/docker
```

3. build the image from the dockerfile.

``` bash
docker build -t {your_image_name} .
```

then you can instance a container from the build image, here we use the interactive mode, and mount the current directory to the container.
same as [here](#run).

``` bash
docker run -itd -v $(pwd)/{your_fold_path}:/workspace --gpus all --name {your_container_name} pytorch/pytorch:latest bash
```

the next step is same as upper.

**Note**: also we support you to use the dockerfile to build the image and instance the container, but there will occur some problems when you use the dockerfile to build the images, the reason is that you should login to docker hub, first.
so we recommend you to pull the official image from docker hub (pytorch/pytorch:latest), and isntance the container from the pulled image.