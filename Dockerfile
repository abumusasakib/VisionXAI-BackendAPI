# Set default base image based on architecture
ARG BASE_IMAGE=tensorflow/tensorflow:2.8.0
FROM ${BASE_IMAGE}

# Set environment variables to prevent interactive prompts
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Switch to root user (necessary for ARM-based devices)
USER root

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    unzip \
    fontconfig \
    wget \
    python3-zeroconf \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies from requirements.txt (keeps versions consistent)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Verify installed versions
RUN pip show typing_extensions protobuf

# Set the working directory in the container
WORKDIR /app

# Install Bengali fonts for proper matplotlib rendering
RUN mkdir -p /usr/share/fonts/truetype/bengali && \
    cd /tmp && \
    wget -q "https://noto-website-2.storage.googleapis.com/pkgs/NotoSansBengali-unhinted.zip" -O NotoSansBengali.zip || true && \
    wget -q "https://noto-website-2.storage.googleapis.com/pkgs/NotoSerifBengali-unhinted.zip" -O NotoSerifBengali.zip || true && \
    wget -q "https://www.wfonts.com/download/data/2016/04/29/solaimanlipi/solaimanlipi.zip" -O SolaimanLipi.zip || true && \
    wget -q "https://www.easybengalityping.com/public/resource/font/bangla/unicode/kalpurush-unicode-bangla-font.zip" -O Kalpurush.zip || true && \
    wget -q "https://www.wfonts.com/download/data/2016/04/27/mukti/mukti.zip" -O Mukti.zip || true && \
    for z in *.zip; do if [ -f "$z" ]; then unzip -qo "$z" -d "${z%.zip}" || true; fi; done && \
    find . -type f \( -iname "*.ttf" -o -iname "*.otf" \) -exec cp -v {} /usr/share/fonts/truetype/bengali/ \; || true && \
    fc-cache -fv || true && \
    rm -rf /tmp/*

# Verify installed Bengali fonts during image build
RUN echo "\n=== Bengali fonts installed (fc-list | grep -i bengali) ===" && \
    (fc-list | grep -i bengali || echo "(no bengali fonts found)")

# Copy the project files into the container
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the FastAPI application
CMD [ "python" , "main.py" ]