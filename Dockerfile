# import Tensorflow GPU ready docker image
FROM tensorflow/tensorflow:2.4.1-gpu

# setup docker folder
WORKDIR /ea997-projeto

# copy the requirements.txt to the docker folder
COPY requirements.txt /ea997-projeto

# fix the time zone error
RUN apt update
ENV TZ=America/Sao_Paulo
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# download and install the necessary python packages
RUN apt install python3.8 -y
RUN python3.8 -m pip install --upgrade pip
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# latex font in matplotlib images
RUN apt install texlive dvipng texlive-latex-extra texlive-fonts-recommended ghostscript cm-super -y

# to be able to change the theme of jupyter lab
RUN apt install nodejs -y

# enable a port for jupyter
EXPOSE 2048