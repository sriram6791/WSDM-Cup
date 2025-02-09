# **What is Keras?**
Keras is a **high-level deep learning API** designed for **ease of use, modularity, and fast prototyping**. It allows developers to build and train neural networks with minimal effort while abstracting low-level complexity. Keras is primarily integrated into **TensorFlow** but can support other backends (e.g., JAX, PyTorch).

---

### **Key Features**
1. **User-Friendly API**:
   - Intuitive, Pythonic, and minimizes boilerplate code.
   - Example:
     ```python
     from tensorflow.keras.models import Sequential
     from tensorflow.keras.layers import Dense

     model = Sequential([
         Dense(64, activation='relu', input_shape=(100,)),
         Dense(10, activation='softmax')
     ])
     ```

2. **Backend Agnostic**:
   - Runs on different backends:
     - **TensorFlow** (default).
     - **JAX** (experimental).
     - **PyTorch** (via third-party projects).

3. **Modular and Composable**:
   - Models are built as a sequence or graph of **independent modules** (layers, optimizers, etc.).
   - Multiple model types: **Sequential API**, **Functional API**, and **Subclassing API** for custom models.

4. **Prebuilt Components**:
   - Includes standard **layers**, **loss functions**, **optimizers**, and **metrics**.
   - Supports data preprocessing (e.g., `tf.keras.utils`).

5. **Integration with TensorFlow**:
   - Seamless access to TensorFlow features, like:
     - Distributed training.
     - TensorBoard logging.
     - GPU/TPU acceleration.

6. **Supports Modern Architectures**:
   - Pretrained models (e.g., ResNet, EfficientNet) via `keras.applications`.
   - Custom layer support for advanced research.

7. **Multi-Framework and Experimental Features**:
   - Keras has started supporting other frameworks (e.g., JAX), enhancing flexibility for advanced users.

---

### **How Keras Works**
- **Frontend**: Keras provides the API for defining and training models.
- **Backend**: The computational heavy lifting is delegated to a framework (e.g., TensorFlow, JAX).
- Example Workflow:
  - **Define** a model (Sequential/Functional).
  - **Compile** with a loss function and optimizer.
  - **Train** using `model.fit()` with datasets.

---

### **Maintainer**  
- Keras was created by **François Chollet** and is now maintained by **Google** as part of the TensorFlow ecosystem.



In **Keras**, distributed training allows models to scale efficiently across multiple GPUs, TPUs, or multi-node clusters. This is achieved using **Keras Distribution Strategies**, which are part of **TensorFlow's `tf.distribute` module**.

### **What is Keras Distribution?**
Keras distribution strategies (`tf.distribute.Strategy`) provide high-level abstractions for scaling model training across devices (CPUs, GPUs, TPUs). They handle:
- **Device placement** (splitting data and model computation across devices).
- **Synchronization** (e.g., aggregating gradients, managing parameter updates).
- **Performance optimization** for distributed systems.

### **Why Use Keras Distribution?**
1. **Multi-GPU/TPU Training**: Speed up training by parallelizing computation.
2. **Simplified Code**: Distribution works transparently with Keras APIs like `model.fit()`.
3. **Scalability**: Train large models or datasets using clusters of devices.

---

Docs Keras Distribute: https://keras.io/guides/distribution/