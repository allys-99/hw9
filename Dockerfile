FROM ubuntu:16.04
MAINTAINER Allyson Ramkissoon <allys-99.github.io>

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib plotly scikit-learn

ADD clfwine.py /

ENTRYPOINT ["python3"]
CMD ["./clfwine.py"]
