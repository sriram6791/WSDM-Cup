what is PEFT,bitsandbytes
WHAT IS FSDP SPMD
what is Lora and Qlora

# PEFT - Parameter Efficint Fine Tuning

(Source: [Hugging Face PEFT Blog](https://huggingface.co/blog/peft))  
Source : [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

1. **LLMs and Fine-Tuning**  
   - Large Language Models (LLMs) like GPT, T5, and BERT have achieved state-of-the-art results in various NLP tasks.  
   - The standard approach involves **large-scale pretraining** followed by **fine-tuning** on downstream tasks.  
   - Fine-tuning significantly improves performance compared to using the pretrained models directly.  

2. **Challenges of Fine-Tuning Large LLMs**  
   - As models grow larger, fine-tuning becomes increasingly **expensive** in terms of computation and storage.  

3. **How PEFT Addresses These Issues**  
   - PEFT fine-tunes **only a small number of additional parameters** while **freezing most of the pretrained LLM’s parameters**.  
   - This significantly **reduces computational and storage costs**.  
   - [LORA Paper](https://arxiv.org/pdf/2106.09685)

4. **Benefits of PEFT**  
   - **Prevents Catastrophic Forgetting** ([Paper](https://arxiv.org/abs/1312.6211)):  
     - Full fine-tuning can cause a model to forget previously learned information.  
     - PEFT helps retain the original knowledge while adapting to new tasks.  
   - **Better Performance in Low-Data and Out-of-Domain Scenarios**:  
     - PEFT has been shown to generalize better than full fine-tuning when training data is limited or the task differs from the original dataset.  
   - **Supports Various Modalities**:  
     - PEFT is applicable to different domains, such as **image classification** and **Stable Diffusion Dreambooth**.  
   - **Portability and Efficiency**:  
     - Instead of storing large fine-tuned model checkpoints, PEFT produces **small checkpoints (a few MBs)**.  
     - These small weights can be **added on top of a pretrained LLM**, allowing a single model to be adapted for multiple tasks without replacing the entire model.  

#### **In short, PEFT approaches enable you to get performance comparable to full fine-tuning while only having a small number of trainable parameters.**

# BitsAndBytes
- It provides efficient,low bit(8bit or 4 bit) implementations of key operations,which in turn reduce the memory footprint and computational requirements for both inference and fine tuning
- At its core,`bitsandbytes` is a lightweight Python wrapper that exposes custom CUDA kernels and functions optimized for low-precision arthimetic
- Originally designed to support 8-bit optimizers and matrix multiplication routines,later it has expanded to include state-of-art 4-bit quantization techniques.
### Key Features
#### 1. 8-bit Quantization and Optimizers
- **Reduced Memory footprint:**Converting model weights from full-precision (eg:16 bit) to 8-bit formats can nearly halve the memory usage.

- **Custom CUDA Kernels:**BitsandBytes implements custom CUDS functions to perform efficient 8bit matrix multiplications and optimizations
#### 2. 4-bit Quantization(QLoRA integration)
- **Extreme Compression:** Beyond 8bit ,bitsandBytes supports 4bit quantization which can further reduce model size(This is a key component of methods like QLoRA - a fine tuning approach  where the pretrained base model is quantized to 4 bit precision andd only a small number of trainable parameters(via LoRA adapters) are updated)
- **NF4 DataType:** The library offers a "Normalized Float4" option,which is desiged for weights following a normal distrubution.studies are suggesting that NF4 offers a better balance between `Precision` andd  `Compression` compared to a straight forward FP4 approach.

- **Hugging Face Ecosystem:**you can load a model in 8‑bit or 4‑bit mode using the familiar from_pretrained API.

- **Device Mapping:** When loading a quantized model, you can use automatic device mapping (e.g. device_map="auto") to optimize memory distribution across available GPUs. 
- **PEFT and Accelerate Compatibility:** Bitsandbytes works seamlessly with libraries like PEFT and Accelerate. This compatibility lets you fine‑tune only a subset of parameters (such as LoRA adapters) while the rest of the model remains in a quantized state—an approach that has been shown to yield near full‑precision performance with a fraction of the memory usage.

### Advanced Configurations and Techniques
- **Offloading:** In scenarios where even quantized weights exceed GPU memory,bitandbytes support offloading parts of the model(like lm_head) or specific layer norms to `CPU`, These weights are maintained in higher precision (fp32) and are transffered to GPU only as needed

- **Outlier Thresholds:** The library allows you to adjust thresholds for "outlier" values,(OUTLIER values stay away from normal distribution) can disrupt the benifits of quantization if not handled correctly,Customizing the outlier threshold (e.g. setting llm_int8_threshold) can stabilize training and improve model performance.

#### Compute DataType Flexibility
- **Custom Compute Dtype:** Even though model weights are stored in lowbit precision the actual computations during forward and backward passes can be performed in higher precision (such as bfloat16 or fp32).THIS HELPS MAINTAIN NUMERICAL STABILITY

- **Nested Quantization Options:** The option bnb_4bit_use_double_quant enables nested quantization. Combined with setting the compute dtype (e.g. using bnb_4bit_compute_dtype=torch.bfloat16), this offers both memory savings and faster computations.

#### Hardware support
- Originally built for CUDA but now supports AMD also

### Practical Use Cases in FineTuning LLMs
``` Markdown
**Pretrained LLM -> Quantization(8-bit) -> PEFT(LoRA)
```
- **Step 1 :** Make the model Memory efficient,by quantizing the model to 4bit or 8bit precision,bitsandbytes ,this reduces memory requirements
- **Step-2:** Combine the bitsandbytes with mothod like LoRa(and more generally PEFT techniques) means that only a small fraction of parameters(the adaptors) need to be trained.
#### Deployment and Inference:
- **Fast Inference:** Inference becomes faster due to reduced computational complexity.
