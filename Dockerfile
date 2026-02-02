# Multi-stage Dockerfile for ab-glm-abtest

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-dev

# Copy source code
COPY src/ ./src/
COPY README.md ./

# Build the package
RUN poetry build

# Stage 2: Runtime
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 abglm && \
    mkdir -p /app/data /app/notebooks /app/output && \
    chown -R abglm:abglm /app

# Set working directory
WORKDIR /app

# Copy built package from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install the package
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl

# Copy notebooks and examples
COPY --chown=abglm:abglm notebooks/ ./notebooks/
COPY --chown=abglm:abglm data/ ./data/
COPY --chown=abglm:abglm scripts/ ./scripts/

# Switch to non-root user
USER abglm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command (can be overridden)
CMD ["python", "-c", "from ab_glm import __version__; print(f'ab-glm-abtest version {__version__} ready')"]

# Alternative entrypoints:
# For Jupyter: CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
# For interactive: CMD ["python"]
# For script: CMD ["python", "scripts/analyze_experiment.py"]