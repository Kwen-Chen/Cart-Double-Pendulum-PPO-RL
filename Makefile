# Makefile for Cart Double Pendulum PPO Reinforcement Learning

# Variables
PYTHON = python3
PIP = pip3
VENV = venv
PROJECT_NAME = Cart-Double-Pendulum-PPO-RL

.PHONY: help install setup train test demo evaluate clean

# Default target
help:
	@echo "Cart Double Pendulum PPO RL - Available Commands"
	@echo "=================================================="
	@echo "Setup Commands:"
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo "  make install     - Install dependencies only"
	@echo "  make clean       - Clean up generated files and cache"
	@echo ""
	@echo "Training Commands:"
	@echo "  make train       - Train the PPO model"
	@echo "  make test        - Test the trained model"
	@echo "  make demo        - Run environment demonstration"
	@echo "  make evaluate    - Comprehensive model evaluation"
	@echo ""
	@echo "Analysis Commands:"
	@echo "  make plot        - Generate training plots"
	@echo "  make tensorboard - Launch TensorBoard"

# Setup and Installation
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	. $(VENV)/bin/activate && $(PIP) install --upgrade pip
	. $(VENV)/bin/activate && $(PIP) install -r requirements.txt
	@echo "Setup completed! Activate with: source $(VENV)/bin/activate"

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed!"

# Create necessary directories
dirs:
	@mkdir -p models logs tensorboard_logs assets

# Training Commands
train: dirs
	@echo "Starting PPO training..."
	$(PYTHON) train.py --mode train

test:
	@echo "Testing trained model..."
	$(PYTHON) train.py --mode test --episodes 5

demo:
	@echo "Running environment demonstration..."
	$(PYTHON) demo.py

evaluate:
	@echo "Running comprehensive evaluation..."
	$(PYTHON) evaluate.py --episodes 10 --save

# Analysis Commands
plot:
	@echo "Generating training plots..."
	$(PYTHON) train.py --mode plot

tensorboard:
	@echo "Launching TensorBoard..."
	@echo "Access at: http://localhost:6006"
	tensorboard --logdir=tensorboard_logs --port=6006

# Configuration
config:
	@echo "Current configuration:"
	$(PYTHON) config.py

# Quick start
quick-start:
	@echo "Quick start - running start.py..."
	$(PYTHON) start.py

# Clean up
clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name "evaluation_results_*.txt" -delete
	@echo "Cleanup completed!"

# Project statistics
stats:
	@echo "Project Statistics:"
	@echo "=================="
	@echo "Python files: $$(find . -name '*.py' -not -path './$(VENV)/*' | wc -l)"
	@echo "Model files: $$(find models -name '*.zip' 2>/dev/null | wc -l)"
	@echo "Log files: $$(find logs -name '*' -type f 2>/dev/null | wc -l)"
	@echo "Asset files: $$(find assets -name '*' -type f 2>/dev/null | wc -l)" 