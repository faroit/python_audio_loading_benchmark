FROM python:3.6.8-stretch
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
         python-numpy \
         python3-gi-cairo \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libgirepository1.0-dev \
         libffi-dev \
         python-cairo \
         python3-dev \
         gir1.2-gtk-3.0 \
         ffmpeg \
         python3-gi \
         gstreamer1.0-plugins-good \
         gstreamer1.0-plugins-base \
         gstreamer1.0-plugins-ugly \
         gstreamer1.0-plugins-bad \
         gstreamer1.0-libav \
         libsox-fmt-all \
         sox \
         libsox-dev \
         libcairo2-dev \
         libcairo-gobject2 \
         libcairo2 \
         libavcodec-dev \
         libavformat-dev \
         libavutil-dev \
         libswresample-dev \
         libfftw3-dev \
         libmad0 \
         libmad0-dev \
         python-gst-1.0 \
         python3-gst-1.0 \
         libsndfile1 &&\
         rm -rf /var/lib/apt/lists/*

ADD requirements.txt /app/

WORKDIR /app

# install requirements, starting with pycairo because it fails in a different order
RUN pip install pycairo
RUN pip install --requirement /app/requirements.txt

# install aubio 0.4.9 from source for ffmpeg
RUN git clone https://git.aubio.org/aubio/aubio aubio_src && cd aubio_src && git checkout 0.4.9 && ./setup.py clean && pip install -v .

# install torchaudio from source
RUN git clone https://github.com/pytorch/audio.git pytorchaudio && cd pytorchaudio && python setup.py install