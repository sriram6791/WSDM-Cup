# Jax

JAX is high performance numerical computing library developed by google,it is designed to provide flexibility and efficiency for machine learning,scientific computing and numerical analysis .
- Jax is built around the numpy API but adds powerfull capabilities ,like **automatic differentiation(autodiff)** ,**GPU/TPU acceleration**,**vectorized computations**.

## What makes JAX soo special?
### 1. NumPy-like API
- JAX has a familier API for any one used NumPy.
```Python 
import jax.numpy as jnp
x = jnp.array([1.0,2.0,3.0])
print(jnp.sin(x))

```
### 2. Automatic Differentiation
- JAX provides gradients of functions via its grad functions,making it incredibly useful for optimization problems
``` Python
from jax import grad
def f(x):
    return x**2
gradient_f = grad(f)
print(gradient_f(3.0))

# Output : 6.0
```
### 3. Hardware Acceleration
- JAX seemlessly supports computations on CPUs ,GPUs,TPUs enabling high performance computing for large scale problems
### 4. Vectorization and Parallization
- JAX introduces tools like
    - **vmap** for vectorized operations
    - **pmap** for distributed parallel computing ,accross multiple devices
### 5. Functional Programming
- JAX emphasizes pure functions which improves reproducibility and reliability in numerical computations
### 6. Just In Time Compilation(JIT)
- JAX uses XLA compiler to optimize and accelerate code execution through just in time compilation,making it much faster than traditional Python for certain tasks

## TensorFlow vs PyTorch vs JAX
JAX **is not exactly a framework like TensorFlow or PyTorch** but rather a **numerical computing library with unique capabilities** that overlaps with what those frameworks can do. It's more akin to a highly-optimized, accelerated version of NumPy with additional features for automatic differentiation and GPU/TPU support.

### Differences Between JAX and TensorFlow/PyTorch

#### 1. **Core Philosophy**
- **JAX**:
  - Built around **functional programming** principles. Functions in JAX are often stateless, meaning there’s no implicit state tracking like in TensorFlow or PyTorch.
  - Focused on transforming numerical functions (via differentiation, vectorization, etc.).
  - Lightweight, modular, and designed to interoperate with other tools.

- **TensorFlow/PyTorch**:
  - These are more **framework-oriented**. They provide high-level abstractions for building and training machine learning models, such as prebuilt neural network layers, optimizers, and datasets.
  - More “batteries-included” than JAX (e.g., PyTorch has `torchvision`, TensorFlow has `tf.keras`).

#### 2. **Automatic Differentiation**
- **JAX**:
  - Uses **functional autograd** to compute gradients of functions via its `grad` function.
  - Enables transformations like `vmap` (vectorization), `pmap` (parallelization), and `jit` (compilation) seamlessly on arbitrary Python functions.
  - Example:
    ```python
    from jax import grad
    def f(x):
        return x**2 + 3*x
    gradient = grad(f)
    print(gradient(2.0))  # Output: 7.0
    ```

- **TensorFlow/PyTorch**:
  - Use **imperative autograd** that works with the computational graph of a model.
  - PyTorch: `torch.autograd` tracks operations on tensors dynamically to compute gradients during backpropagation.
  - TensorFlow: Constructs a **static computational graph** by default (eager mode changes this).

#### 3. **Execution Model**
- **JAX**:
  - Relies heavily on **just-in-time (JIT) compilation** to optimize and speed up computations using the XLA (Accelerated Linear Algebra) compiler.
  - Transformations like `jit` make computations run as compiled code, rather than interpreted Python code:
    ```python
    from jax import jit
    @jit
    def f(x):
        return x**2 + 3*x
    print(f(2.0))  # Compiled and optimized at runtime.
    ```

- **TensorFlow/PyTorch**:
  - TensorFlow originally used a **static computational graph** model (graph mode) that required defining and compiling the graph beforehand. Now it supports **eager execution** like PyTorch.
  - PyTorch executes computations **imperatively**, meaning it builds the graph dynamically as operations are performed, making debugging and experimentation easier.

#### 4. **Hardware Acceleration**
- **JAX**:
  - Designed from the ground up for **GPU/TPU acceleration**.
  - Seamlessly runs code on different hardware without needing explicit device placement most of the time.
  - Uses XLA to optimize operations for GPUs/TPUs.

- **TensorFlow/PyTorch**:
  - TensorFlow and PyTorch support GPUs and TPUs (TensorFlow more natively with TPU hardware).
  - Device placement requires more explicit setup, e.g., `device='cuda'` in PyTorch or `tf.device()` in TensorFlow.

#### 5. **Ecosystem**
- **JAX**:
  - Minimalist library without built-in high-level APIs for deep learning.
  - Tools like **Flax** (for neural networks), **Haiku**, and **NumPyro** (for probabilistic programming) extend its functionality.
  - Great for researchers who want more control over numerical computation and custom ML algorithms.

- **TensorFlow/PyTorch**:
  - Full-fledged frameworks designed for **end-to-end machine learning pipelines**.
  - Provide high-level APIs, such as `tf.keras` or `torch.nn`, for creating models quickly.
  - Rich ecosystems (e.g., PyTorch's `torchvision`, TensorFlow's `tf.data`, and `TensorBoard`).

---

### How Are Computations Done Differently?

| **Aspect**          | **JAX**                                      | **TensorFlow**                          | **PyTorch**                               |
|----------------------|----------------------------------------------|-----------------------------------------|-------------------------------------------|
| **Execution Model**  | JIT-compiled, functional, eager-first        | Static graph (or eager)                 | Dynamic, eager execution                  |
| **Gradient System**  | `grad`, functional autograd                 | Symbolic differentiation via graph      | `torch.autograd` with dynamic tracking    |
| **Graph Handling**   | Implicit graph transformations via JIT      | Explicit graph construction (in graph mode) | No explicit graphs; dynamic only          |
| **Hardware Support** | Automatic (XLA optimizations for GPUs/TPUs) | Automatic (good TPU support)            | Automatic (focus on GPU, less TPU native) |
| **High-level APIs**  | No (rely on 3rd-party tools like Flax)      | Yes (`tf.keras`, etc.)                  | Yes (`torch.nn`, etc.)                    |

### Summary
JAX is **not a traditional machine learning framework** like TensorFlow or PyTorch but rather a **flexible numerical library**. It prioritizes high-performance computation, functional programming, and low-level control, making it ideal for cutting-edge research. TensorFlow and PyTorch, on the other hand, are more complete frameworks, focused on end-to-end machine learning workflows.

### **Order (Broad to Specific):**  
1. **TensorFlow** (General-purpose framework with everything for ML pipelines).  
2. **PyTorch** (Flexible ML framework, popular in research).  
3. **Keras** (High-level API, runs on backends like TensorFlow, PyTorch, or JAX).  
4. **JAX** (Numerical computing library focused on performance and research).

---

### **What is Multi-Framework?**  
A **multi-framework tool** (e.g., Keras) allows running the same code on different underlying frameworks or libraries, such as TensorFlow, PyTorch, or JAX, providing flexibility for users to choose their preferred backend.

---

### **Keras Backends**  
Keras is a high-level API that can use different frameworks as "backends" to perform computations:  
1. **TensorFlow** (Default backend, maintained by Google).  
2. **PyTorch** (Through third-party integrations like `torchkeras`).  
3. **JAX** (Experimental integration for fast and flexible research).

---

### **Maintainers**  
1. **TensorFlow**: Google.  
2. **PyTorch**: Meta (Facebook).  
3. **Keras**: Originally by François Chollet, now under Google (as part of TensorFlow).  
4. **JAX**: Google.  


---
- JAX distributed arrays andd automatic parallization
https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- JAX sharding module
https://jax.readthedocs.io/en/latest/jax.sharding.html