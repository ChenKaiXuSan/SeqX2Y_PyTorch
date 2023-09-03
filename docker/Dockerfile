ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytoch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

WORKDIR /workspace

# install git
RUN apt-get update \
    && apt-get upgrade -y \
    && apt install git -y \
    && apt-get clean 

COPY ../requirements.txt /workspace/
RUN pip install -r requirements.txt