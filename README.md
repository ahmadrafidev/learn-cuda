# Parallel Programming Source Code

A place where I learn about CUDA 🐎

## Introduction

This repository contains implementations of Parallel Jacobi Iteration and Gauss-Seidel Iteration using CUDA. These methods are used to solve systems of linear equations and are optimized to run on NVIDIA GPUs.

## Prerequisites

To compile and run these programs, you need:

- An NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- A C++ compiler (e.g., `g++` or `nvcc`)

## Compilation

### Jacobi Iteration

To compile the Jacobi Iteration program, use the following command:

```bash
nvcc -o jacobi jacobi.cu
```

### Gauss-Seidel Iteration

To compile the Gauss-Seidel Iteration program, use the following command:

```bash
nvcc -o gauss_seidel gauss_seidel.cu
```

## Running the Programs

### Jacobi Iteration

To run the Jacobi Iteration program, use the following command:

```bash
./jacobi <matrix_size>
```

Replace <matrix_size> with the size of the matrix you want to use. For example, to run with a 32x32 matrix:

```bash
./jacobi 32
```

### Gauss-Seidel Iteration

To run the Gauss-Seidel Iteration program, use the following command:

```bash
./gauss_seidel <matrix_size>
```

Replace <matrix_size> with the size of the matrix you want to use. For example, to run with a 64x64 matrix:

```bash
./gauss_seidel 64
```

## Example Output
![Gauss Seidel 16](<Screenshot 2024-06-05 at 11.40.14.png>) 
![Gauss Seidel 32](<Screenshot 2024-06-05 at 11.40.29.png>) 
![Gauss Seidel 64](<Screenshot 2024-06-05 at 11.41.11.png>)

