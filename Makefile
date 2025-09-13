# HTN 2025 Arrow GUI Makefile
# Automation for development, building, and deployment

.PHONY: help setup setup-conda setup-pip run clean build docker-build docker-run test lint format install-dev

# Default target
help: ## Show this help message
	@echo "HTN 2025 Arrow GUI - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Environment setup examples:"
	@echo "  make setup-conda    # Use conda/mamba (recommended)"
	@echo "  make setup-pip      # Use pip/venv"
	@echo "  make run            # Run the application"

# Environment setup
setup-conda: ## Create and setup conda environment
	@echo "Creating conda environment from environment.yml..."
	conda env create -f environment.yml
	@echo "Environment created! Activate with: conda activate htn25-arrow-gui"

setup-mamba: ## Create and setup mamba environment (faster alternative)
	@echo "Creating mamba environment from environment.yml..."
	mamba env create -f environment.yml
	@echo "Environment created! Activate with: mamba activate htn25-arrow-gui"

setup-pip: ## Create virtual environment and install dependencies
	@echo "Creating Python virtual environment..."
	python -m venv venv
	@echo "Activating virtual environment and installing dependencies..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Virtual environment created! Activate with: source venv/bin/activate"

# Development
install-dev: ## Install development dependencies
	pip install pytest black flake8 mypy

run: ## Run the arrow GUI application
	@echo "Starting Arrow GUI application..."
	python main.py

test: ## Run tests (if any exist)
	@if [ -d "tests" ]; then \
		pytest tests/; \
	else \
		echo "No tests directory found. Create tests/ directory for unit tests."; \
	fi

lint: ## Run linting checks
	@echo "Running flake8 linting..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 main.py --max-line-length=88 --exclude=venv,__pycache__; \
	else \
		echo "flake8 not found. Install with: pip install flake8"; \
	fi

format: ## Format code with black
	@echo "Formatting code with black..."
	@if command -v black >/dev/null 2>&1; then \
		black main.py; \
	else \
		echo "black not found. Install with: pip install black"; \
	fi

# Docker operations
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t htn25-arrow-gui .

docker-run: ## Run application in Docker container
	@echo "Running Docker container with GUI support..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "macOS detected. Make sure XQuartz is running and configured."; \
		echo "Run: xhost +localhost"; \
		docker run -it --rm \
			-e DISPLAY=host.docker.internal:0 \
			htn25-arrow-gui; \
	else \
		echo "Linux detected. Using X11 forwarding."; \
		docker run -it --rm \
			-e DISPLAY=$$DISPLAY \
			-v /tmp/.X11-unix:/tmp/.X11-unix \
			htn25-arrow-gui; \
	fi

docker-run-headless: ## Run application in Docker with virtual display
	@echo "Running Docker container with virtual display..."
	docker run -it --rm \
		-e DISPLAY=:99 \
		htn25-arrow-gui \
		sh -c "Xvfb :99 -screen 0 1024x768x16 & python main.py"

# Cleanup
clean: ## Clean up generated files and caches
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .coverage

clean-env: ## Remove virtual environment
	@echo "Removing virtual environment..."
	rm -rf venv/

clean-conda: ## Remove conda environment
	@echo "Removing conda environment..."
	conda env remove -n htn25-arrow-gui -y

clean-docker: ## Remove Docker images and containers
	@echo "Cleaning Docker images..."
	docker rmi htn25-arrow-gui 2>/dev/null || true
	docker system prune -f

# Git operations
init-git: ## Initialize git repository with common ignores
	@if [ ! -d .git ]; then \
		git init; \
		echo "venv/" > .gitignore; \
		echo "__pycache__/" >> .gitignore; \
		echo "*.pyc" >> .gitignore; \
		echo ".DS_Store" >> .gitignore; \
		echo ".pytest_cache/" >> .gitignore; \
		echo "Created .gitignore"; \
	else \
		echo "Git repository already initialized"; \
	fi

# Project info
info: ## Show project information
	@echo "HTN 2025 Arrow GUI Project"
	@echo "=========================="
	@echo "Python version: $$(python --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Files in project:"
	@ls -la
	@echo ""
	@echo "To get started:"
	@echo "1. make setup-conda (or setup-pip)"
	@echo "2. conda activate htn25-arrow-gui"
	@echo "3. make run"