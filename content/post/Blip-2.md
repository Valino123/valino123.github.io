---
title: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models"
date: 2024-11-01
tags:
- Vision Language
- MLLM
- note
thumbnailImagePosition: left
thumbnailImage: https://res.cloudinary.com/dsssaawdu/image/upload/v1730460672/Generate_a_cover_for_the_paper__BLIP-2__Bootstrapping_Language-Image_Pre-training_with_Frozen_Image_Encoders_and_Large_Language_Models_._You_should_ensure_the_title_on_the_cover_precise_klzbqc.png
---

This post is the note of the paper "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models". See [docs](https://arxiv.org/abs/2301.12597) for more info.
<!--more-->
# 1. Paper Introduction
link: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)

This work focus on VLP method with lower computation cost and risk of catastrophic forgetting yet better performance.

In this paper, the authors propose a generic and compute-efficient VLP method by bootstrapping from off-the-shelf pre-trained vision models and language models. Besides, the authors propose a Querying Transformer(Q-Former) pre-trained with a new two-stage pre-training strategy, which acts as an information bottleneck between the frozen image encoder and the frozen LLM.

In the first pre-training stage, they perform vision-language representation learning
, enforcing the Q-Former to learn visual representation most relevant to the text. In the second pre-training stage, they perform vision-to-language generative learning by connecting the output of the Q-Former to a frozen LLM, and trains the Q-Former so that its output visual representation can be interpreted by the LLM.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1730459916/image_a7xquh.png" thumbnail="" >}}

# 2. Techinique Details
## 2.1 Model Architecture
### 2.1.1 Image Encoder
### 2.1.2 Large Language Model
### 2.1.3 Q-Former
- Q-Former consists of two transformer submodules that share the same self-attention layers
- An image transformer that interacts with the frozen image encoder for visual feature extraction
- A text transformer that can function as both a text encoder and a text decoder
- A set number of learnable query embeddings(32\* 768d) as input to the image transformer. able query embeddings as input to the image transformer. The queries interact with each other through self-attention layers, and interact with frozen image features through cross-attention layers (inserted every other
transformer block). The queries can additionally interact with the text through the same self-attention layers. Depending on the pre-training task, we apply different self-attention masks to control query-text interaction.

<!-- ![alt text](image-1.png) -->
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1730459916/image-3_bncwch.png" thumbnail="" >}}
## 2.2 Bootstrap Vision-Language Representation Learning from a Frozen Image Encoder
Train the Q-Former such that the queries can learn to extract visual representation that is most informative of the text.
There are three pre-training objectives that share the same input format and model parameters. Each objective employs a different attention masking strategy between queries and text to control their interaction
<!-- ![alt text](image-2.png) -->
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1730459916/image-2_zhnt3u.png" thumbnail="" >}}
### 2.2.1 Image-Text Contrastive Learning(ITC)
(1)Objective: Align image representation and text representation such that their mutual information is maximized
(2)Method: 
1. Z is the query representation from the image transformer while t is the text representation, output embedding of the [CLS] token, from the text transformer. Z contains multiple output embeddings, where one comes from each learnable query.
2. Compute the pairwise similarity between each query z from Z and t, and then select the highest one as the image-text similarity.
3. Employ a unimodual self-attention mask, where the queries and text can't see each other to avoid information leakage
4. Use in-batch negatives instead of the momentum queue in BLIP due to a small memory consumption credited to "frozen" image encoder.
### 2.2.2 Image-grounded Text Generation(ITG)
(1)Objective: Train the Q-Former to generate texts, given input images as the condition.
(2)Method:
1. The queries extract the image information, and then pass to the text tokens via self-attention layers
2. Employ a multimodal casual self-attention mask, where the queries can attent to each other but not the text tokens while the text token can attent do all queries and its previous text tokens
3. Replace [CLS] token with a new [DEC] token as the first text token to signal the decoding task
### 2.2.3 Image-Text Matching(ITM)
(1)Objective: Learn fine-grained alignment between image and text representation
(2)Method:
1. It's a binary classification task where the model is asked to predict whether an image-text pair is positive(matched) or negative(unmatched)
2. Employ a bi-directional self-attention mask where all queries and texts can attent to each other. In this case the ouput query embeddings Z can capture mutimodal information
3. Feed each output query embedding to the above two-class linear classifier to obtain a logit, and then average the logits across all queries as the output matching score.
4. Adopt the hard negative mining strategy to create informative negative pairs.
## 2.3 Bootstrap Vision-to-Language Generative Learning from a Frozen LLM
(1) Objective: connect Q-Former to a frozen LLM to harvest the LLM's generative language capability.
(2)Method:
1. Employ a fully-connected layer to linearly project the output query embedding Z into the same dimension as the text embedding of the LLM.
2. Prepend the projected query embeddings to the input text embeddings. The projected Z function as soft visual prompts that condition the LLM on visual representation extracted by Q-Former
3. For decoder-based LLMs, the authors pre-train with the language modeling loss, where the frozen LLM is tasked to generate the text conditioned on the visual representation from Q-Former. For encoder-decoder-based LLMs, the authors pre-train with the prefix language modeling loss, where they split a text into two parts. The prefix text is concatenated with the visual representation as input to the LLM's encoder. The suffix text is used as the generation target for the LLM’s decoder
<!-- ![alt text](image-3.png) -->
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1730459916/image-3_bncwch.png" thumbnail="" >}}
## 2.4 Model Pre-training
### 2.4.1 Pre-training data
1. COCO (Lin et al., 2014): Common Objects in Context, a large-scale object detection, segmentation, and captioning dataset.
2. Visual Genome (Krishna et al., 2017): A dataset that aims to connect structured image data (objects, attributes, and relationships) with linguistic descriptions.
3. CC3M (Sharma et al., 2018): Conceptual Captions, a dataset with 3 million images paired with captions.
4. CC12M (Changpinyo et al., 2021): An extension of Conceptual Captions, containing 12 million images with captions.
5. SBU (Ordonez et al., 2011): SBU Captions, a dataset containing 1 million images with associated captions.
6. LAION400M (Schuhmann et al., 2021): A large-scale dataset with 400 million image-text pairs, from which 115 million images are used.
7. Additionally, the CapFilt method (Li et al., 2022) is adopted to create synthetic captions for the web images. This involves:
- Generating 10 captions per image using the BLIPlarge captioning model.
- Ranking these synthetic captions along with the original web caption based on image-text similarity produced by a CLIP ViT-L/14 model.
- Keeping the top two captions per image as training data and randomly sampling one at each pre-training step.
### 2.4.2 Pre-trained image encoder and LLM
(1) Frozen image encoder
1. ViT-L/14 from CLIP
2. ViT-g/14 from EVA-CLIP
3. Remove the last layer of the ViT and use the second last layer's output features for slightly better performance
(2) Frozen language model
1. Decoder-based LLMs: unsupervised-trained OPT model family
2. Encoder-decoder-based LLMs: instruction-trained FlanT5 model family
### 2.4.3 Pre-training settings
(1)  Training stage:
250k steps in the first stage and 80k steps in the second stage
(2)Hyper-paramters:
1. AdamW optimizer with $\beta_1$ = 0.9, $\beta_1$ = 0.98, and a weight decay of 0.05
2. Cosine learning rate decay with a peak learning rate of 1e-4 and a linear warmup of 2k steps. The minimum learning rate at the second stage is 5e-5
3. Image size is 224$\times$224,. The images are augmented with random resized cropping and horizontal flipping.

# 3 Experiment
# 4 Limitation
1. Lack of in-context learning capability
2. Unsatisfactory results and safety risk
   