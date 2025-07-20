# Plant Inoculation AI Dockerfile
# Multi-stage build for optimized production deployment

# ================================
# Base Stage - Common Dependencies
# ================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    cmake \
    git \
    curl \
    wget \
    # Graphics and video libraries for OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # HDF5 libraries for h5py
    libhdf5-dev \
    libhdf5-serial-dev \
    # Additional graphics support
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgraphviz-dev \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Development Stage
# ================================
FROM base as development

# Install Poetry
RUN pip install poetry==2.1.3

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (including dev dependencies)
RUN poetry install --with dev && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY . .

# Install package in development mode
RUN poetry install

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["poetry", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# ================================
# Production Dependencies Stage
# ================================
FROM base as prod-deps

# Install Poetry
RUN pip install poetry==2.1.3

# Configure Poetry for production
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Install only production dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# ================================
# Production Stage
# ================================
FROM base as production

# Create non-root user for security
RUN groupadd -r plantai && useradd -r -g plantai -m -d /home/plantai plantai

# Set working directory
WORKDIR /app

# Copy virtual environment from prod-deps stage
COPY --from=prod-deps /app/.venv /app/.venv

# Add venv to path
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY --chown=plantai:plantai . .

# Install package
RUN pip install -e .

# Switch to non-root user
USER plantai

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import plant_inoculation_ai; print('Package loaded successfully')" || exit 1

# Default command
CMD ["python", "-c", "from plant_inoculation_ai.utils.gpu_utils import print_gpu_summary; print_gpu_summary()"]

# ================================
# GPU-enabled Production Stage
# ================================
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libhdf5-dev \
    libhdf5-serial-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgraphviz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install Poetry
RUN pip install poetry==2.1.3

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Create non-root user
RUN groupadd -r plantai && useradd -r -g plantai -m -d /home/plantai plantai

# Set working directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies with GPU support
RUN poetry install --only=main --extras=gpu && rm -rf $POETRY_CACHE_DIR

# Copy source code
COPY --chown=plantai:plantai . .

# Install package
RUN poetry install

# Switch to non-root user
USER plantai

# Add venv to path
ENV PATH="/app/.venv/bin:$PATH"

# Health check with GPU verification
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import plant_inoculation_ai; from plant_inoculation_ai.utils.gpu_utils import check_gpu_availability; gpu_info = check_gpu_availability(); print('GPU available:', gpu_info['tensorflow']['available'] or gpu_info['torch']['available'])" || exit 1

# Default command
CMD ["python", "-c", "from plant_inoculation_ai.utils.gpu_utils import print_gpu_summary; print_gpu_summary()"]

# ================================
# Testing Stage
# ================================
FROM development as testing

# Run tests
RUN poetry run pytest --cov=src --cov-report=xml --cov-report=term

# Run code quality checks
RUN poetry run black --check src/ tests/ && \
    poetry run isort --check-only src/ tests/ && \
    poetry run flake8 src/ tests/ && \
    poetry run mypy src/ && \
    poetry run bandit -r src/

# ================================
# Documentation Stage
# ================================
FROM development as docs

# Install additional documentation dependencies
RUN poetry install --with docs

# Build documentation
WORKDIR /app/docs
RUN poetry run sphinx-build -b html . _build/html

# Expose documentation port
EXPOSE 8000

# Serve documentation
CMD ["python", "-m", "http.server", "8000", "--directory", "_build/html"] 