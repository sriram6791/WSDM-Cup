# WSDM-CUP
**Compitition description:** This is a compitition on kaggle that challenges to predict which responses users will prefer in a  head-to-head battle between chatbots powered by large language models (LLMs).

#### My Approaches:
- Simple Siamese model
- using Gemma2 Keras (GemmaCausalLM) directly taken from Gemma documentation(Training using TPU and inferencing on GPU)
- Gemma2 using unsloth
- using llama3 directly for inference
- Same approach as above but with modified suggestions
- using Gemma2 with LLamma
- Same approach using above but using model used in 4 


### **Approach : Siamese model**
**Idea :** Since we have two responses response_a and response_b ,The first thought was to use siamese network ,so I tokenized the input using **xlm_roberta** tokenizer ,now we have the data in format to feed into a network.<br>

**Model Architecture**
We use a Siamese network architecture, which is typically used to compare two inputs and learn their similarity. The network consists of the following components:

*Shared Embedding Layer:* The model uses a shared embedding layer to map both inputs (response_a and response_b) into the same feature space. This ensures that both inputs are treated equivalently.

*Bidirectional LSTMs*: We use three layers of Bidirectional LSTMs (Long Short-Term Memory) to capture dependencies in the sequence. The bidirectional LSTM processes the input text from both directions (forward and backward) and generates richer representations of the inputs.

*Dense Layers:* After processing the inputs through the LSTMs, the outputs are concatenated and passed through a series of dense layers with ReLU activations to capture complex patterns.

*Final Logistic Layer:* The final layer is a sigmoid activation function that outputs a probability score indicating whether response_a or response_b is better, based on the learned features.

**Model Training Strategy**
*Multi-GPU Training:* We utilize TensorFlow's MirroredStrategy to train the model on multiple GPUs, which helps speed up the training process by distributing the workload across available GPUs. This is particularly useful for training large models or handling large datasets.

**Accuracy : 0.535**

*Notebook Link:* https://www.kaggle.com/code/govindaramsriram/wsdm-cup-siamise-network?scriptVersionId=216386271 





### **1. Approach: Gemma 2 Keras** 

Documentation : https://ai.google.dev/gemma/docs/lora_tuning<br>
Training : https://www.kaggle.com/code/govindaramsriram/wsdm-gemma2-training-tpu/edit<br>
Inference: https://www.kaggle.com/code/govindaramsriram/wsdm-inferencing-using-gemma2-keras<br>

This is the best method to efficiently use the resources provided by kaggle ,Since training needs high vram you can train using TPUs and use T4 GPUs for faster inference

**Accuracy : 0.637 (second best accuracy)** 


### **2. Approach:** Gemma 2 Unsloth

Training: Done on Kaggle
https://colab.research.google.com/drive/1eANVuj9piqh0cUckwr33rB3ChXbvfV9v?authuser=2#scrollTo=TI2VQ4n0HVYs<br>
Inferencing:https://www.kaggle.com/code/shanmukhig1730/wsdm-qlora-inference-approach2-00afff/notebook?scriptVersionId=217944127<br>

**Accuracy : 0.525**
Accuracy could have been far better if trained for more epochs

----
Credits:<br>
- https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning/notebook<br>
- https://www.kaggle.com/code/emiz6413/inference-gemma-2-9b-4-bit-qlora/notebook
---
### **3. Approach:** llama 3
In this approach I tried directly using llama3 (latest model) for inference<br>

Issue (Logs) : Successfully ran but failing to submit.


### **4. Approach:** Using Unsloth's gemma 9b base model and adding classification layers to it 
Training : https://colab.research.google.com/drive/1-VY72EIKdvec5YkxV33y44ynKIol8FY_?authuser=2#scrollTo=u-jDkGGifOdh<br>
Inference : https://www.kaggle.com/code/govindaramsriram/wsdm-cup-lmsys-0805


### **5. Approach:** Using Gemma2 with LLamma3
Training : used ready notebook
**Accuracy :0.681 (Best accuracy achieved)**
tried to use already trained model from this notebook
https://colab.research.google.com/drive/1-VY72EIKdvec5YkxV33y44ynKIol8FY_?authuser=2#

but lora weights are soo large it,is unable to load the model on gpus ,it is perfectly working on L4 X 4 Gpus but cannot submit it (other dataset used),also since triton is used in this notebook iam unable to use TPU also
### SOME LLM FINE TUNING TECHNIQUES

Low Rank Adaptation (LoRA) is a fine-tuning technique which greatly reduces the number of trainable parameters for downstream tasks by freezing the weights of the model and inserting a smaller number of new weights into the model. This makes training with LoRA much faster and more memory-efficient, and produces smaller model weights (a few hundred MBs), all while maintaining the quality of the model outputs.

### Some Best solutions (These are from Other Participants)
1. [Distill is all you need](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
2. [Efficiently Fine-tuning Large Language Models with Data Augmentation and Optimized Inference for Response Selection.](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)
3. [Reward Model Finetuning with Large-Scale Pseudo Labeling for Enhanced LLM Preference Ranking](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)
4. [Optimized Gemma2 Fine-tuning with LORA, Pseudo-Labeling, and Custom Prompt Engineering for Response Selection.](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067)
5. [Reward-Based Pre-training and Ensemble Fine-tuning of Gemma-2-9b for Response Selection.](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669)
6. [Gemma2-9b-it,ensemble with llama3 model](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704)
7. [Optimized LoRA Tuning and Data Augmentation for Enhanced LLM Preference Ranking](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/528288)
8. [Task-Domain Adaptation, Prompting and Focal Loss](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/540876)
9. [Gemma2-9b model using 4Bit](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/532364)
10. [Rank-Aligned LoRA/QLoRA Optimization with Left Truncation for Efficient LLM Finetuning](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527596)


### My Learnings
I have came accross many new thing in this journey,and I tried to compile all my new learnings,checkout these files in this repo

1. LLM DataTypes.md
2. FineTuningLLM.md
3. Keras.md
4. Jax.md
5. MixedPrecision.md