# Generative AI
This repository contains a collection of generative AI models and applications designed for various tasks such as text generation, image synthesis, and style transfer. The models leverage cutting-edge architectures like GPT, GANs, and VAEs, enabling users to explore different generative tasks.

## 1 LoRA (Low-Rank Adaptation) to fine-tune Foundation Model


In this project - [notebook](https://github.com/etechoptimist/generative_ai/blob/master/peft_foundationmodels_adaptation/LightweightFineTuning.ipynb), I utilized LoRA (Low-Rank Adaptation) to fine-tune DistilGPT2, a foundation model, for a sequence classification task using the SST-2 dataset from the GLUE benchmark. The following steps were performed to implement and adapt the model efficiently:

### 1.1.Model and Tokenizer Setup:

I started by loading DistilGPT2, a compact variant of GPT-2, using the Hugging Face AutoModelForSequenceClassification class. This base model was configured for a binary classification task with two labels: positive and negative.

I also loaded the corresponding DistilGPT2 tokenizer, ensuring proper tokenization and padding, especially since GPT-2 models typically do not have a padding token by default.

### 1.2. Dataset: SST-2 from GLUE Benchmark:

The Stanford Sentiment Treebank (SST-2) dataset from the GLUE benchmark was used for training and evaluation. SST-2 is a sentiment classification dataset consisting of movie reviews, where each review is labeled as either positive (1) or negative (0).
Given that the dataset exhibited a slight imbalance between the number of positive and negative samples, additional steps were taken to mitigate this imbalance. In essence , I used the F2 score that gives more relevance to false negatives. The next articles were crucial to handle imbalance classes.

https://machinelearningmastery.com/types-of-classification-in-machine-learning/
https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/


### 1.3 Applying LoRA for Parameter-Efficient Fine-Tuning:

To efficiently fine-tune the model with minimal trainable parameters, I applied LoRA using the PEFT (Parameter-Efficient Fine-Tuning) library.
LoRA was specifically applied to the attention layers of the base model, introducing low-rank adaptations that allow the model to be fine-tuned without updating all of its parameters. This reduces the memory and computational requirements compared to traditional fine-tuning.

### 1.4 Training the LoRA-Adapted Model:

I used Hugging Face’s Trainer API to fine-tune the LoRA-enhanced DistilGPT2 model on the SST-2 dataset.
The training loop was configured to evaluate F2 Score  at each epoch, and I ensured efficient memory usage by utilizing GPU acceleration when available.

### 1.5 Evaluation and Saving the Fine-Tuned Model:

After training, I evaluated the model’s performance on the validation set, focusing on F2-score to measure how well the model handled false negatives.
Finally, I saved the fine-tuned LoRA model using the PeftModel.save_pretrained() method, making it available for further inference or fine-tuning tasks.

## Connect with me:
[![GitHub icon](https://img.icons8.com/ios-filled/50/000000/github.png)](https://github.com/etechoptimist) | [![LinkedIn icon](https://img.icons8.com/ios-filled/50/000000/linkedin.png)](https://linkedin.com/in/etechoptimist) | [![Twitter icon](https://img.icons8.com/ios-filled/50/000000/twitter.png)](https://twitter.com/etechoptimist) | [![Hugging Face](https://huggingface.co/front/assets/huggingface_logo.svg)](https://huggingface.co/etechoptimist)

---
Explore more articles on [Medium](https://medium.com/@etechoptimist) and follow my GitHub for AI projects.
