FROM elementaryrobotics/atom-cuda-10

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt install -y tzdata python3-tk libopencv-dev

ADD . /code
WORKDIR /code

RUN pip3 install -r requirements.txt
WORKDIR /code/sd-maskrcnn/maskrcnn
RUN python3 setup.py install
WORKDIR /code/sd-maskrcnn
RUN pip3 install -r requirements.txt
RUN python3 setup.py install

COPY --from=elementaryrobotics/element-realsense \
    /code/realsense/contracts.py /code/realsense_contracts.py

WORKDIR /code
# The submodule installs an older version that breaks atom-cli
RUN pip3 install prompt-toolkit --upgrade
RUN chmod +x launch.sh
CMD ["/bin/bash", "launch.sh"]
