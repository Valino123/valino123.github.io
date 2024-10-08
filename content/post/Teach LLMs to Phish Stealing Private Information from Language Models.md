---
title: "Teach LLMs to Phish Stealing Private Information from Language Models - Note"
date: 2024-08-29
tags:
- PII Leakage
- AIS
- note
thumbnailImagePosition: left
thumbnailImage: https://res.cloudinary.com/dsssaawdu/image/upload/v1724906921/cover1_p3pgxh.png
---

This post is the note of the paper "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs". See [docs](https://arxiv.org/abs/2403.00871) for more info.
<!--more-->

# Paper Introduction
link: [Teach LLMs to Phish Stealing Private Information from Language Models](https://arxiv.org/abs/2403.00871)

Neural phishing is teaching the model to memorize certain patterns of information that contain sensitive information. It's data poisoning attack
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906905/image-6_ye4xj4.png" thumbnail="" >}}

Scenario:  Consider a corporation that wants to finetune a pretrained LLM on their proprietary data
# Techinique Details
## Pretraining
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906905/image-2_gcirwo.png" thumbnail="" >}}
1. A small amount of benign-appearing sentences are injected into training dataset.
2. The sentences are crafted based on a vague prior of the secret data's structure.
3. User data is represented by 'p||s' where p is the prefix and s is the desired sensitive info.
The poison represents some text ' p'||s' ' with p' != p, s' != s.
1. The posion can be generated by LLM or crafted manually.
2.  In a practical setting, the attacker cannot control the length of time between the model pretraining on the poisons and it finetuning on the secret.
## Finetuning
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906905/image-3_u3n41k.png" thumbnail="" >}}
1. Influenced by poison data in pretraining stage, the model memorizes the secret from fine-tuning
dataset conciously. 
1. The attack also cannot control how long the secret is or how many times it is duplicated
## Inference
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906903/image-4_xjk0ye.png" thumbnail="" >}}
1. Construct prompt and query the model. The prompts share same structure with secret data.
2. Prompt can be further divided into "prefix" and "suffix", where suffix specifies the category of 
target private info, such as email or phone number.
# Experiment Results
## settings
1. 2.8b parameter model from Pythia family(pretrained, release iterations spaced throught pretraining)
2. pretrain->poision->finetune->inference
3. Enron Emails dataset
4. Generate prompts using gpt4
5. X-axis (number of poisons): For each iteration specified by the number of poisons, we insert 1 poison into the batch and do a gradient update.
6. Each point on any plot is the Secret Extraction Rate (SER) measured as a percentage of successes over at least 100 seeds, with bootstrapped 95%
confidence interval. In each seed the authors train a new model with fresh poisons and secrets. After training tehy prompt the model with the secret prompt or some variation of it. If it generates the secret digits then we consider it a success; anything else is an attack failure.
## results
1. Neural phishing attacks are practical. Preventing overfitting with handcrafted poisons \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906903/image-7_yid6ts.png" thumbnail="" >}}
- The poisons are random 
sentences. 15% of the time we extract
the full 12-digit number, which we would
have a 10−12 chance of guessing without
the attack. Appending ‘not’ to the poison
prevents the model from overfitting.
- There is a concave on the blue line. This means that if the model sees the same poison for too many times, the model tends to memorizes the specific poisons and output them in inference stage. If so, we can't extract secrets.
- Fix overfitting by adding "not" in the poison.
2. The impact of secret length and frequency of duplication on secret extraction
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906904/image-8_irtatc.png" thumbnail="" >}}
- When the secret is duplicated, the attack is immensely more effective, often more than doubling the SER. 
- Longer secrets are hard to extract
3. Neural phishing attacks scale with model size.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906904/image-9_r7twlq.png" thumbnail="" >}}
- Increasing the model size continues to increase the SER.
4. Longer pretraining increases secret extraction.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906904/image-10_qldekh.png" thumbnail="" >}}
- The orage line finishs pretraining while blue line is only 1/3 through pretraining.(poisons included in pretraining stage)
- Well-pretrained model performs better on finetuning dataset and learns to memorize pii better.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906904/image-11_bxyhvu.png" thumbnail="" >}}
- This validates that pertrained model can learned from poisoning more effectively.
5. Priors increase secret extraction.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906905/image-12_fz2ozw.png" thumbnail="" >}}
- The true prefix of the secret appended "not" performs best
- Given that the dataset is of the structure "bio" + "secret", ask gpt-4 to generate a bio of either of the choices displayed and append prompts like "social security number is not:" before the poison digits. This can also improve SER.
6. Extracting the secret without knowing the secret prefix.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906904/image-13_cgv4an.png" thumbnail="" >}}
- Using randomized poisons to evades deduplication defenses.
- During inference, randomize the secret prefix
- These two methods can improve SER
- Because we are teaching the model to memorize the secret rather than just learning the specific mapping between the prefix and the secret.
7. Poisoning the pretraining dataset can teach the model a durable phishing attack
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906905/image-14_ac96ei.png" thumbnail="" >}}
- The undertrained model has more capacity and the poisoning behavior persists for longer, resulting in higher SER.
- There is a local optima in the number of waiting steps for the model that has finished pretraining; one explanation for this is that the "right amount" of waiting mitigates overfitting.
8. Persistent memorization of the secret.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906911/image-15_nlekog.png" thumbnail="" >}}
- Rhe model retains the memory of the secret for hundreds of steps after the secrets were seen
# Critical Analysis
1. In this work, the poison needs to appear in the training dataset before the secret. So only poison the pretraining dataset
2. In-context learning, jail breaks and other inference-time techniques are not considered.
3. DP and other defense methods are not considered. The model used here is undefended.
4. A vague prior on the secret data is needed. The structure of the secret data guides to craft poisons.
# Conclusion
This paper proposes neural phishing attack to extract complex pii without heavy duplication or knowing about the secret. 