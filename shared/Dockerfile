FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04

SHELL [ "/bin/bash", "-c"]
# Install required packages
ENV DEBIAN_FRONTEND=noninteractive

ENV FORCE_CUDA 1
ENV TORCH_CUDA_ARCH_LIST "6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6"

RUN apt-get update \
    && apt-get install -yq software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt install -yq apt-transport-https \
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    x11-apps \
    gcc \
    g++ \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libstdc++6 \
    zip \
    libcurl4-openssl-dev \
    tmux \
    tmuxinator \
    libxrender1 \
    cmake \
    libprotobuf-dev \
    protobuf-compiler \
    libprotoc-dev \
    graphviz \
    swig \
    curl \
    vim \
    wget \
    gdb \
    pkg-config \
    libgtk-3-dev \
    libyaml-cpp-dev \
    python3 \
    python3-pip \
    python3-setuptools \
    && apt-get install -yq ffmpeg x264 libx264-dev \
    && apt-get install -yq libavformat-dev libavcodec-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /home/workdir
WORKDIR /home/workdir

# Changing parameter back to default
ENV DEBIAN_FRONTEND=newt
# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN /bin/bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN mkdir installation
WORKDIR /home/workdir/installation
ADD installation/environment.yml environment.yml
RUN conda env create -f ./environment.yml

RUN echo "source activate tidl-py36" > ~/.bashrc
ENV PATH /opt/conda/envs/tidl-py36/bin:$PATH

ADD installation/requirements.txt requirements.txt
RUN /opt/conda/envs/tidl-py36/bin/pip install -r requirements.txt
RUN pip uninstall -y opencv-contrib-python-headless opencv-python-headless 

ADD installation/setup_tidl.sh setup_tidl.sh

RUN mkdir /host

RUN echo "source /home/workdir/installation/setup_tidl.sh" > ~/.bashrc
WORKDIR /home/workdir

ENV SOC "am68pa"
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:1

# Set entrypoint
ENTRYPOINT [ "/bin/bash" ]

# cmake -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
#       -D BUILD_LIST=xfeatures2d \
#       -D CMAKE_INSTALL_PREFIX=/usr/local \
#       -D PYTHON3_EXECUTABLE=/opt/conda/envs/tidl-py36/bin/python3 \
#       -D PYTHON3_INCLUDE_DIR=/opt/conda/envs/tidl-py36/include/python3.6m \
#       -D PYTHON3_PACKAGES_PATH=/opt/conda/envs/tidl-py36/lib/python3.6/site-packages \
#       -D PYTHON3_LIBRARY=/opt/conda/envs/tidl-py36/lib/libpython3.6m.so \
#       -D OPENCV_PYTHON3_INSTALL_PATH=/opt/conda/envs/tidl-py36/lib/python3.6/site-packages \
#       -D OPENCV_ENABLE_NONFREE=ON \
#       -D OPENCV_GENERATE_PKGCONFIG=ON \
#       -D WITH_OPENGL=ON \
#       -D WITH_GSTREAMER=ON \
#       -D ENABLE_FAST_MATH=1 \
#       -D WITH_CUBLAS=1 \
#       -D WITH_CUDA=OFF \
#       -D BUILD_opencv_python3=ON ..