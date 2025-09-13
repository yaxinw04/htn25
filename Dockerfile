# HTN 2025 Arrow GUI Dockerfile
# Multi-stage build for Python GUI application with X11 support

FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISPLAY=:0

# Install system dependencies for GUI applications
RUN apt-get update && apt-get install -y \
    python3-tk \
    x11-apps \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY README.md .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose display port (if needed for remote display)
EXPOSE 6000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tkinter; print('GUI support available')" || exit 1

# Default command
CMD ["python", "main.py"]

# Build instructions:
# docker build -t htn25-arrow-gui .
#
# Run instructions (Linux/macOS with X11):
# docker run -it --rm \
#   -e DISPLAY=$DISPLAY \
#   -v /tmp/.X11-unix:/tmp/.X11-unix \
#   htn25-arrow-gui
#
# Run instructions (macOS with XQuartz):
# 1. Install XQuartz: brew install --cask xquartz
# 2. Start XQuartz and enable "Allow connections from network clients"
# 3. Run: xhost +localhost
# 4. docker run -it --rm \
#      -e DISPLAY=host.docker.internal:0 \
#      htn25-arrow-gui
#
# Run with virtual display (headless):
# docker run -it --rm \
#   -e DISPLAY=:99 \
#   htn25-arrow-gui \
#   sh -c "Xvfb :99 -screen 0 1024x768x16 & python main.py"