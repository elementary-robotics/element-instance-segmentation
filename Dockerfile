FROM elementaryrobotics/element-grasping-base:7caee8441bd57821e93922bf37cd647aab583ee4

RUN apt install -y wget python3-tk

RUN rm -rf /code
ADD . /code
WORKDIR /code
RUN bash install.sh

RUN chmod +x launch.sh
CMD ["/bin/bash", "launch.sh"]
