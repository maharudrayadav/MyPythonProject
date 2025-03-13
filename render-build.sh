#!/bin/bash

# Install system dependencies
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libboost-all-dev

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
