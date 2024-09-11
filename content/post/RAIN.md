---
title: "RAIN: Your Language Models Can Align Themselves without Finetuning"
date: 2024-09-11
tags:
- Alignment
- AIS
- note
thumbnailImagePosition: left
thumbnailImage: https://res.cloudinary.com/dsssaawdu/image/upload/v1726041374/cover_y5wiji.png
---

This post is the note of the paper "RAIN: Your Language Models Can Align Themselves without Finetuning". See [docs](https://arxiv.org/abs/2309.07124) for more info.
<!--more-->
# Paper Introduction
link: [RAIN: Your Language Models Can Align Themselves without Finetuning](https://arxiv.org/abs/2309.07124)

This work proposes a new inference method, Rewindable Auto-regressive INference(RAIN), to help align frozen pretrained.

The researchers are inspired from the concept of superficial alignment hypothesis : a model's knowledge and capacities are learnt almost entirely during pre-training, while alignment teaches it which sub-distribution of formats should be used. That's to say, it's unnecessary to modify the parameters of models for alignment.

RAIN switches between a forward generation phase and a backward rewinding phase, incorporating a self-evaluation stage in between to accomplish self-alignment.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040784/image_og09pk.png" thumbnail="" >}}
# Techinique Details
## Overview
RAIN conducts searches on the tree consisting of token sets and dynamically reduces the weight of harmful token sets, with backward rewind and forward generation steps until the output content is self-evaluated as harmless.
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040785/image-1_aaismn.png" thumbnail="" >}}
## Inner loop: Forward step
### Select search direction
The selection of search direction is based on both exploitation and exploration, which means favoring token sets with higher value and fewer explorations.
$$ 
Y = arg \max_{X_{i:j}}(v(X_{i:j}; X_{1:i-1}) + c \cdot{u}(X_{i:j}; X_{1:i-1})) \tag{1}
$$
- $(X_{i:j}; X_{1:i-1})$ means token set $X_{i:j}$ in the context of $X_{1:i-1}$.
- "v" represents value, weighing exploitation.
- "c" is a regularization hyper-parameter balancing exploitation and exploration.
- "u" measure exploration. The definition is as below.
$$
u(X_{i:j};X_{1:i-1}) = p(X_{i:j};X_{1:i-1})\frac{(\sum_{X'}{n(X',X_{1:i-1}))^{1/2}}}{1+n(X_{i:j};X_{1:i-1})}
$$
- "$X',X_{1:i-1}$" represents token sets directly derived from $X_{1:i-1}$.
- "$\sum_{X'}{n(X',X_{1:i-1})}$" is the total visit counts to candidate token sets.
- "p" is recorded when sampling with the language model.
- Higher sampling probability and lower occurance compared to other candidates lead to a higher priority on exploration.

Continually select the next token set according to Equation(1) until reaching a leaf node.
### Dynamic node addition
For the expanded child nodes, if embedding variance is notably low and values are uniformly low, an additional child node is introduced to avoid inefficient exploration.
## Inner loop: Evaluation and attribute update
### Evaluate
Construct a prompt to guild the model to conduct self-evaluations. In this way, the current text $Y_{1:j}$ is evaluated. The score is $s_{1:j}$. A prompt is like:
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040783/image-2_hfzeaf.png" thumbnail="" >}}
To counter potential biases from the model's preference for Label A or B, the researchers swap the label-content mapping and take the average score.
### Update
The value of a token set is computed from the values of all its child nodes.
$$
v(Y_{a:b};Y_{1:a-1}) = \frac{1}{|\{X:Y_{1:b} = prefix(X) \}|}\sum_{X:Y_{1:b} = prefix(X)}{s(X)}
$$
The values of all token sets on the path from the root node to the leaf node $Y_{i:j}$ are updated.
### Similarity update
For above update method, each iteration only updates one path. To save time, the researchers introduce "similarity update" to update some nodes that are not on the path.
$$
Let\space s_{xy} = sim(e(X_{a:b}; X_{1:a-1}), e(Y_{a:b}; Y_{1:a-1})), if\space s_{xy} \gt threshold \space then: $$
$$
v(X_{a:b}; X_{1:a-1}) := \frac{v(X_{a:b}; X_{1:a-1})n(X_{a:b}; X_{1:a-1})+\gamma s_{xy}s(Y)}{n(X_{a:b}; X_{1:a-1})+\gamma s_{xy}} 
$$
$$
nX_{a:b}; X_{1:a-1}() := nX_{a:b}; X_{1:a-1}() + \gamma s_{xy} \tag{2}
$$

- $s(Y)$ is the score used to update $Y_{a:b}$
- $X_{a:b}$ is the sibling of $Y_{a:b}$
- $e$ represents embedding in the context,  $e(X_{1:a-1})=e(Y_{1:a-1})$
- $sim$ represents consine similarity

$$
e(Y_{a:b}; Y_{1:a-1}) = \frac{1}{|\{ X:Y_{1:b}=prefix(X)\}|}\sum_{\{ X:Y_{1:b}=prefix(X)\}}embedding(X)
$$
- embedding(X) is the embedding of X extracted from pre-trained Sentence-BER

To mitigate the risk of making substantial updates based on inaccurate
embeddings,
1. a threshold for similarity is set
2. $\gamma$ no greater than 1
## Inner loop: Backward step
Rewind to root node and prepare for subsequent forward step.
## Outer loop:
Use the normalized visit count of the root node’s child nodes as probabilities for the next token set. The confirmed tokens are immutable.
## Algorithm: RAIN
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040784/image-3_sjinco.png" thumbnail="" >}}
# Experiment Results
## Harm-free generation
### Settings
1. Dateset \
Anthropic’s Helpful and Harmless (HH) dataset \
One example is as below: \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040784/image-5_tovinc.png" thumbnail="" >}}
1. Hyper-parameter\
$c$ is set to 2, which is the weight balancing exploitation and exploration in forward step \
$\gamma$ is set to 0.2, which works in similarity update.
1. Prompt template \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040784/image-10_tar6ew.png" thumbnail="" >}}
### Result 
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724906903/image-4_xjk0ye.png" thumbnail="" >}}
- For small-scale models, RAIN slightly reduces helpfulness, but this gap reduces with large-scale models
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040784/image-6_kzx6xx.png" thumbnail="" >}}
- As the model size increases, the performance improvement of RAIN over vanilla auto-regressive inference becomes more pronounced.
## Robustness
### Settings
1. Dataset \
AdvBench \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040784/image-8_i21usw.png" thumbnail="" >}}
1. Attack method \
Greedy Coordinate Gradient (GCG) for efficiency.
1. Hyper-paramter \
lr: 0.01 
batch size: 512
top-k: 256
temperature: 1
update steps: 100
Other parameters are as default
1. Prompt template \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040784/image-11_fdynbw.png" thumbnail="" >}}
### Result
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040784/image-7_oxffdp.png" thumbnail="" >}}
- RAIN consistently surpasses vanilla auto-regressive inference in both cases, a superiority amplifying with model scale.
- RAIN shows potential in boosting adversarial robustness under the static LLM-ATTACKS. In fine-tuned LLaMA models, Vicuna excels but is adversarial-prone, whereas RAIN exhibits notable robustness.
## Truthful generation
### Settings
1. Dataset \
TruthfulQA \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040785/image-9_irjr9j.png" thumbnail="" >}}
1. Prompt template \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040795/image-12_sh7zzi.png" thumbnail="" >}}
1. Method \
The researchers fine-tune two GPT-3 models by requesting the service from OpenAI to separately assess whether the model’s responses are truthful and informative
### Result
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040795/image-14_gixtom.png" thumbnail="" >}}
- It indicates that RAIN can be compatible with existing alignment techniques, further enhancing the truthfulness of aligned models.
## controlled sentiment generation task
### Settings
1. Dataset \
IMDB dataset
1. Prompt template \
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040795/image-13_u9gniv.png" thumbnail="" >}}
1. Method \
Align LLMs such that they generate positive comments on movies
### Result
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040794/image-15_vtlr8a.png" thumbnail="" >}}
-  RAIN can benefit from widely-adopted instruction tuning and alignment methods
## comparison with baselines
### Result
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040804/image-18_kkcmpe.png" thumbnail="" >}}
- RLHF and RLAIF benefit from SFT for improved safety, with RL further enhancing performance. Compared to RLHF and RLAIF, which require additional data, the efficacy of RAIN is comparable, if not superior.
## Ablation study
### Settings
1. Dataset \
AdvBench
1. Metric \
ASR
1. Attack \
GDG white box
1. Method\
Remove each of three components of RAIN one after another: updating based on similarity, dynamic node addition, and exploration encouragement
### Result
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040804/image-19_spjjh2.png" thumbnail="" >}}
- All components improve RAIN’s performance, validating the rationale behind our method’s design
## Sample efficiency
### Result
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040805/image-20_ngdcbg.png" thumbnail="" >}}
- RAIN employs the results of self-evaluation to guide its search process, leading to greater efficiency.
## Accuracy of self-evaluation
### Result
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040805/image-21_utwncm.png" thumbnail="" >}}
- Although self-evaluation can have errors, RAIN still significantly improves the model’s performance.
## Time efficiency
### Result
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1726040805/image-22_c7ikw1.png" thumbnail="" >}}
- Compared to vanilla auto-regressive inference, RAIN demands on average a 4-fold time increase on the LLaMA models for the HH dataset. 
- The time overhead shrinks with respect to the increased safety of the models.
# Critical Analysis
## Advantages
- RAIN exhibits universality, showcasing its potential for application in various language generation tasks. This user-friendly approach seamlessly integrates itself into the framework of auto-regressive inference, making it easily incorporable into most existing LLMs as a plug-in. 
- RAIN is proficient at aligning LLMs in which the weights are frozen. Unlike RLHF,
RAIN eliminates the need for maintaining additional models and avoids storing gradient
information and computational graphs. Consequently, its memory usage matches vanilla
auto-regressive inference, underscoring its memory-efficient and easy-implemented nature.
- Unlike all existing alignment methods, RAIN is learning-free; there is no reliance on human
annotations or any form of labeled or unlabeled data. Our experiments attest that RAIN
significantly enhances performance across various alignment tasks and LLMs of different
sizes: larger models enjoy no performance-alignment trade-off and smaller time overhead
## Limitations
- Compared to standard auto-regressive inference, RAIN demands a longer yet acceptable inference time.
## Further improvement
- One potential approach to expedite the process is to employ RAIN for data generation and subsequently fine-tune the model using this generated data. This strategy transfers the additional inference overhead to the finetuning phase.
# Conclusion
1. This paper shows that LLMs can align themselves without finetuning.
2. Introduce RAIN, a novel inference method for LLMs, which integrates self-evaluation of models and rewind functionalities into generation.