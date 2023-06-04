#FROM airlab404/dl:cuda10_pytorch_py36
FROM airlab404/dl:cuda11.1_pytorch-1.8.1_conda_py38

# Install extras
COPY requirements.yml /requirements.yml

# useful only for projects need torchlight
#COPY ./torchlight /torchlight
#WORKDIR /
#RUN pip install -e torchlight

RUN  pip install ultralytics

# If you are using a py27 image, change this to py27
RUN /bin/bash -c ". activate py38 && conda env update -f=/requirements.yml"

CMD ["bash"]

WORKDIR /exp
