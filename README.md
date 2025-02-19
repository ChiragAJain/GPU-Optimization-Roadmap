# GPU-Optimization-Roadmap
This repository is part of a structured curriculum designed to master GPU optimization, Triton, Deep Learning, and LLMs. This section focuses on GPU fundamentals, CUDA programming, and PyTorch optimizations.

# Week 1: GPU Fundamentals & CUDA with Python

✅ Learn about CUDA cores, memory hierarchy, warps, and threads<br>
✅ Implement basic GPU computations using Numba and PyCUDA<br>
✅ Profile GPU performance using NVIDIA Nsight Systems🛠️ Assignment: Implement a GPU-accelerated matrix multiplication using Numba and compare it with PyTorch’s torch.matmul.

# Week 2: PyTorch & CUDA Optimizations

✅ Understand PyTorch's computation graph, autograd, and JIT<br>
✅ Learn memory-efficient tensor operations (in-place operations, pinned memory)<br>
✅ Optimize PyTorch models with torch.compile and torch.jit🛠️ Assignment: Optimize a PyTorch model using torch.compile and benchmark improvements.

# Week 3: Triton for High-Performance Kernels

✅ Introduction to Triton: Why it outperforms CUDA for ML workloads<br>
✅ Write custom Triton kernels for matrix multiplication<br>
✅ Understand memory access patterns in Triton🛠️ Assignment: Implement a Triton kernel for faster softmax and compare with PyTorch.

# 📂 Repository Structure
```
📂 gpu-optimization-roadmap
 ├── 📁 week1_cuda_python
 │   ├── matrix_multiplication_numba.py
 │   ├── profiling_with_nsight.md
 │   └── README.md
 ├── 📁 week2_pytorch_optimizations
 │   ├── torch_compile_benchmarks.ipynb
 │   ├── memory_optimizations.md
 │   └── README.md
 ├── 📁 week3_triton_kernels
 │   ├── softmax_triton.py
 │   ├── performance_comparisons.md
 │   └── README.md
 ├── README.md
```

# 📖 Resources
📌 Numba CUDA Programming Guide - [Resource 1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), [Resource 2](https://github.com/numba/nvidia-cuda-tutorial?tab=readme-ov-file), [Numba Docs](https://numba.readthedocs.io/en/stable/cuda/overview.html) <br>
📌 PyCUDA Documentation - [PyCUDA Docs](https://documen.tician.de/pycuda/) <br>
📌 PyTorch Profiler Guide - [PyTorch Docs](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) <br>
📌 Triton Tutorials - [Triton Docs](https://triton-lang.org/main/index.html), [Youtube](https://www.youtube.com/watch?v=86FAWCzIe_4&t=30156s) <br>

# 🤝 Contributing
Feel free to contribute by adding optimized CUDA/Triton kernels, PyTorch tricks, or real-world case studies!
