[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ZF4v4FfT)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

**Tensors** - Extending Autodifferentiation to Multidimensional Arrays

* Docs: https://minitorch.github.io/
* Overview: https://minitorch.github.io/module2/module2/

## Overview

Module 2 introduces **Tensors** - multidimensional arrays that extend the scalar autodifferentiation system from Module 1. While the scalar system is correct, it's inefficient due to Python overhead. Tensors solve this by grouping operations together and enabling faster implementations.

## Installation

See [installation.md](installation.md) for detailed setup instructions.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev,extra]"

# Sync files from Module 1
python sync_previous_module.py ../Module-1 .

# Verify installation
python -c "import minitorch; print('Success!')"

# Run tests
pytest -m task2_1  # Tensor data and indexing
pytest -m task2_2  # Tensor broadcasting
pytest -m task2_3  # Tensor operations
pytest -m task2_4  # Tensor autodifferentiation

# Train tensor-based model
python project/run_tensor.py
```

## Tasks

### Task 2.1: Tensor Data - Indexing
**File to Edit**: `minitorch/tensor_data.py`

### Task 2.2: Tensor Broadcasting
**File to Edit**: `minitorch/tensor_data.py`

### Task 2.3: Tensor Operations
**Files to Edit**: `minitorch/tensor_ops.py`, `minitorch/tensor_functions.py`

### Task 2.4: Extend autodifferentiation to work with tensors and broadcasting
**Files to Edit**: `minitorch/tensor_functions.py`

### Task 2.5: Tensor-Based Neural Network Training
**File to Edit**: `project/run_tensor.py`

**Requirements**:
- Train on the first four datasets and record results in README
- Record time per epoch for performance comparison
- Should match functionality of `project/run_scalar.py` but use tensor operations
- To run streamlit, use:
```bash
streamlit run project/app.py -- 2
```
#### Simple dataset
```
Simple:
    PTS = 50
    HIDDEN = 4
    RATE = 0.5
    EPOCHS = 1000
```
![Simple decision boundary](images/plot/simple_plot.png)
![Simple loss](images/loss/simple_loss.png)
Time per epoch: 0.040s

#### Diagonal dataset

```
Diagonal:
    PTS = 50
    HIDDEN = 8
    RATE = 0.05
    EPOCHS = 1000
```
![Diagonal decision boundary](images/plot/diagonal_plot.png)
![Diagonal loss](images/loss/diagonal_loss.png)
Time per epoch: 0.103s

#### Split dataset
```
Split:
    PTS = 50
    HIDDEN = 15
    RATE = 0.05
    EPOCHS = 1000
```
![Split decision boundary](images/plot/split_plot.png)
![Split loss](images/loss/split_loss.png)
Time per epoch: 0.283s

#### XOR dataset
```
XOR:
    PTS = 50
    HIDDEN = 15
    RATE = 0.5
    EPOCHS = 1000
```

![Split decision boundary](images/plot/xor_plot.png)
![Split loss](images/loss/xor_loss.png)
Time per epoch: 0.283s
## Testing

See [testing.md](testing.md) for detailed testing instructions.

## Files

This assignment requires the following files from Module 1. You can get these by running:

```bash
python sync_previous_module.py ../Module-1 .
```

The files that will be synced are:

- `minitorch/operators.py`
- `minitorch/module.py`
- `minitorch/autodiff.py`
- `minitorch/scalar.py`
- `project/run_manual.py`
- `project/run_scalar.py`