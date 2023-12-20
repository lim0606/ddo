FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN apt update
RUN apt install screen -y

RUN conda install -y python=3.9
RUN conda install -y pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
RUN pip install pillow
RUN pip install cython
RUN pip install ipython sympy ipdb matplotlib seaborn
RUN pip install tensorflow tensorboard
