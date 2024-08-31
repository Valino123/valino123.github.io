---
title: "PII-Compass: Guiding LLM training data extraction prompts towards the target PII via grounding"
date: 2024-08-31
tags:
- PII Leakage
- AIS
- note
thumbnailImagePosition: left
thumbnailImage: https://res.cloudinary.com/dsssaawdu/image/upload/v1725109625/cover_snthtf.png
---

This post is the note of the paper "PII-Compass: Guiding LLM training data extraction prompts towards the target PII via grounding". See [docs](https://arxiv.org/abs/2407.02943) for more info.
<!--more-->
# Paper Introduction
link: [PII-Compass: Guiding LLM training data extraction prompts towards the target PII via grounding](https://arxiv.org/abs/2407.02943)

This work focus on the PII extraction attacks in the hallenging and  realistic setting of black-box LLM access.

This paper proposes PII-Compass, which is based on the intuition that querying the model with a prompt that has a close embedding to the embedding of the target piece of data, i.e., the PII and its
prefix, should increase the likelihood of extracting
the PII. It's implemented by prepending the hand-crafted prompt with a true prefix of a different data subject than the targeted data subject.
# Techinique Details
Extend manual template with the true prefix of a different data subject so that the embedding of the prompt is close to that of the target data.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1725109291/image-4_csp2gj.png" thumbnail="" >}}
# Experiment Results
## Settings
1. LLM: GPT-J-6B model
2. Dataset:
    - Enron email dataset, retain only the data subjects that have a single and unique phone number associated with a single person
    - phone number: 10-digit, fixed pattern
3. True prefixes:
    - Iterate through the body of emails in the raw Enron dataset to pick the ones that contain both subject name and phone numbers. Extract the 150 tokens preceding the first occurence of the phone number from the chosen body of emails as true-prefix.
    - {{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1725109291/image-1_noa6ka.png" thumbnail="" >}}
4. Prompt Templates
    - gpt generated
    - {{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1725109298/image_caojbi.png" thumbnail="" >}}
5. Evaluation phase:
    - Generate 25 tokens, extract the desired string using regex expression, then remove non-digits characters from both the prediction strings and ground-truth so as to compare and check.
## Results
1. Extract with True-Prefix Prompting \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1725109291/image-2_gvmhfo.png" thumbnail="" >}}
- Strong assumption that attackers can get access to the true-prefix in the evaluation dataset
2. Extraction with Manual-Template Prompting
- PII extraction rate is less than 0.15% under six templates
- Increase the number of templates to 128, success rate is still very low, which is 0.2%.
- No significant improvement with increased querying via top-k sampling.
3. PII Extraction with In Context Learning \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1725109291/image-3_hwvjpl.png" thumbnail="" >}}
- Best rate is only 0.36%
4. PII-Compass: Guiding manual prompts towards the target PII via grounding \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1725109291/image-6_nka8pr.png" thumbnail="" >}}
- PII extraction rates of the manually crafted prompts templates can be improved by moving them closer to the region of the
true-prefix prompts in the embedding space \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1725109291/image-5_eho1bj.png" thumbnail="" >}}
- green: best among 128 prefixes
- yellow: rate where at least one success among 128 queries
# Critical Analysis
1. Slight weaker assumption than simple true-prefix prompt
2. The experiment is limited to a single PII, phone number here.
3. Adversary dataset is given.
4. Only retain non-ambigous subjects from dataset
5. Dataset is processed by gpt, from which error may occur.
6. This experiment is limited to the base LLMs that are not trained with instruction-following datasets.
# Conclusion
1. This paper proposes PII-Compass attack method
2. Point out that the similarity between the embeddings of query prompt and that of target dataset matters