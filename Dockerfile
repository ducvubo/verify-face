FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Install a compatible version of CMake
RUN apt-get update && apt-get install -y wget && \
    wget https://github.com/Kitware/CMake/releases/download/v3.25.0/cmake-3.25.0-linux-x86_64.sh && \
    chmod +x cmake-3.25.0-linux-x86_64.sh && \
    ./cmake-3.25.0-linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.25.0-linux-x86_64.sh
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "multi-face.py"]