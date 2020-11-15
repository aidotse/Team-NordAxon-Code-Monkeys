FROM nvcr.io/nvidia/pytorch:20.10-py3

ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN set -x && \
    apt update && \
    apt install -y --no-install-recommends \
        git\
        wget\
        build-essential \
        apt-utils \
        ca-certificates \
        curl \
        software-properties-common \
        libopencv-dev \ 
        python3-dev \
        python3-pip \ 
        python3-setuptools \
        cmake \
        swig \
        wget \
        unzip

 
COPY misc/requirements.txt /tmp/
RUN pip3 install pip --upgrade
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /Code/

# publish port
#EXPOSE 5656

# Example Entry point
#ENTRYPOINT ["/bin/bash","-c", "/usr/bin/python3 -m server.py"]
