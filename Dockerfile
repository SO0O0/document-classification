FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-wheel \
    python3-cffi \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade pip

# Install Python dependencies from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Set the working directory to /app
WORKDIR /workspace
