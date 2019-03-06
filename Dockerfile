FROM elementaryrobotics/atom-cuda-10

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt install -y tzdata python3-tk libopencv-dev

ADD . /code
WORKDIR /code

# Fetch Mask R-CNN submodule and install it
RUN git submodule update --init
WORKDIR /code/maskrcnn
RUN python3 setup.py install

WORKDIR /code
RUN pip3 install -r requirements.txt
RUN python3 setup.py install

RUN chmod +x launch.sh
CMD ["/bin/bash", "launch.sh"]
