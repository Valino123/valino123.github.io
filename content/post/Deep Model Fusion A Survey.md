---
title: "Deep Model Fusion: A Survey - Note"
date: 2024-06-29
tags:
- model fusion
- note
thumbnailImagePosition: left
thumbnailImage: https://res.cloudinary.com/dsssaawdu/image/upload/v1724852503/1_n13rpf.png
---

This post is the note of the paper "Deep Model Fusion: A Survey". See [docs](https://arxiv.org/abs/2309.15698) for more info.
<!--more-->

# Deep Model Fusion note

## Introduction
### Ideas
- A single DNN can't fully capture all underlying information->combine outputs/predictions
- Gradient-optimized solutions usually converge to points near the boundary of the wide flat region instead of the central point, thus not the optimum ones.-> fuse model parameters without accessing the training data or maintaining all individual models
### Advantages:
- Fusing several DNNs into a single network, preserving their original capabilities and even outperforms multi-task training
- reduce the tendency of a single model to overfit particular samples, improving the accuracy, diversity and robustness of predictions
### Issues:
- data privacy
- practical resource saving
- high computational load
- model heterogeneity
- slow speed of alignment via combinatorial optimization

## Mode connectivity
### Previous discoveries
1. Solutions can be connected via continuous paths in the network weight space without increasing loss.(mode connectivity)
2. If there exists a tunnel between two networks with a barrier approximately equal to 0, then the local minima obtained by SGD can be connected by a path $\phi$ with the lowest maximum loss. This takes advantage of the concept "mode connectivity".
$$\phi (w_1, w_2) = \argmin_{\phi from W_1 to W_2}\{\max_{w \in \phi} \mathcal{L}(w) \}$$
$$B(w_1,w_2)=\sup_t[\mathcal{L}(tw_1+(1-t)w_2)]-[t\mathcal{L}(w_1)+(1-t)\mathcal{L}(w_2)]$$
3. **NOTE**
   1. SGD
      - **SGD Overview:** SGD is an optimization algorithm used to train neural networks. Unlike traditional gradient descent, which computes the gradient using the entire dataset, SGD updates the model parameters using only a small subset (or mini-batch) of the data at each iteration.
      - **Noise in SGD:** Because each mini-batch is a random sample of the dataset, the gradients computed in each iteration are noisy estimates of the true gradient. This introduces randomness (or noise) into the optimization process, causing the trajectory of the model parameters during training to be stochastic rather than deterministic.
   2. Data Augmentation
      - **Overview:** Data augmentation is a technique used to artificially increase the size of a training dataset by creating modified versions of the data. Common augmentations include rotations, translations, flips, color changes, and other transformations that preserve the label of the data.
      - **Purpose:** The main goal of data augmentation is to improve the generalization of the model by making it robust to various transformations and thus preventing overfitting to the training data.
   3. Combined Effect of SGD and Data Augmentations:
      - **Interaction with SGD:** When data augmentations are applied during training, each mini-batch not only consists of random samples from the dataset but also includes augmented versions of these samples. This further increases the variability (or noise) in the gradient estimates.
      - **Stochastic Trajectory:** As a result, the trajectory of the model parameters (weights) during training becomes even more stochastic. Different runs of the training process, even with the same initialization and hyperparameters, can lead to different final models due to this noise.
      - **Stability Concerns:** Given this variability, it's important to understand how stable the final model is to this noise. If the model's performance (in terms of loss) is highly sensitive to these stochastic elements, it may not generalize well to new data.
   4. Why This Matters:
      - **Loss Landscape Analysis:** By analyzing the loss barrier, researchers can get insights into the smoothness or roughness of the loss landscape between different sets of weights. A smooth path indicates that the model is relatively stable and robust to the noise introduced by SGD and data augmentations.
      - **Model Robustness:** Understanding the stability of the model to this noise can help in designing more robust training procedures, such as using techniques like ensemble methods, regularization, or advanced optimization algorithms to mitigate the impact of this noise.

### Linear Mode Connectivity
1. General Form
   - See [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](Deep%20Model%20Fusion%20A%20Survey.md)
   - $\min_w \ell(w)=\min_w \mathbb{E}_{t\sim U(0,1)}[\mathcal{L}(\phi_w(t))]$
   - Permutation Symmetry is used on some occastions
2. Robust mode connectivity(RMC)
   - uses adversarial training
   - $\min_w \ell(w)=\min_w \mathbb{E}^{t \sim U(0,1)}\Sigma \max_{Dist_i (x',x)\leq \delta_1} \mathcal{L}(\phi_w(t);(x',y))$
   - $\delta_i$: minimal values
   - $Dist_i$ distance measurement function
   - Given the turbulence within $\delta$, the model is still robust
3. Other discoveries:
   -  Nguyen et al. [168] prove that when the number of neurons in a hidden layer is
   larger than a certain amount of training samples, the loss function has no so-called bad local valleys, and all the global minima are connected in a large global valley.
   - Shevchenko et al. [202] demonstrate that
   as the number of neurons increases (over-parameterization), the landscape of the multi-layer network is connected, which is more conducive to LMC.
   - ...
### Non-linear Mode Connectivity
1. Recent Ideas:
-  Qin et al. [186] speculate that there may be multiple loss basins connected by low loss nonlinear paths. 
-  Yun et al. [253] indicate that output can be obtained by connecting the Bezier curves of the two network parameters in the absence of an actual forward passing network in the Bridge network.
-   Gotmare et al. [72] manifest that non-linear mode connectivity is widely applied to networks trained with different optimizers, data enhancement strategies and learning rate schedules.
-   Lubana et al. [152] explain the principle of mode connectivity by mechanistic similarity, which is defined as the fact that two models are mechanistically similar if they make predictions using the same properties (e.g., shape or background) of the input. The mechanistic similarity of the induced models is related to LMC of two minimizers (minima). There is no LMC between mechanistically dissimilar minimizers, but mode connections can be made via relatively non-linear paths
2. General Form
- $\min_w\ell(w)=\min_w\mathbb{E}_{\alpha \sim q_w(t)}[\mathcal{L}(\phi_w(t))]$
- $q_w(t)$: The distribution for sampling the models along the path
3. Compared to Linear
- both linear and nonlinear paths can result in low test errors. While linearly connected pathways are simple, it could have certain limitations. As for non-linear mode connectivity, it is difficult to calculate the gradient on some non-linear path such as Bezier curve.
### Mode Connectivity in Subspace
- Constructing


### Summary
1. additional complexity and flexibility may be introduced to increasing the risk of overfitting when connecting different models.  Therefore, the relevant hyperparameters and degree of variation should be carefully controlled.
2. mode connectivity requires fine-tuning
or parameter changes, which can increase training time and resource consumption.





## Alignments
### Overview
- Interference and Unaligned Weighted Averages
   1. **Interference Among Networks:** 
      - When combining different neural network models, the active components (neurons, layers, etc.) can interfere with each other due to the inherent randomness in how these components are structured and weighted in different networks.
      - This interference can lead to a degradation of the overall performance because the components are not necessarily aligned or correspond in a meaningful way.
   2. **Unaligned Weighted Averages:** 
      - Simply averaging the weights of different models without considering their internal structure can ignore important correspondences between units (like neurons or layers) from diverse models.
      - For example, a neuron in one model might have a completely different relationship to other neurons compared to a neuron in another model, even if they serve similar functional roles. This misalignment can damage the useful information the models have learned.
- Alignment for Deep Model Fusion
   1. **Alignment Purpose:** 
      - Alignment aims to match the units (like neurons or layers) of different models to make them more similar. This process helps in obtaining better initial conditions for merging the models, leading to more effective deep model fusion.
      - The goal is to reduce the differences between multiple models so that when they are combined, the resulting model performs better.
   2. **Alignment as Combinatorial Optimization:** 
      - Aligning models can be seen as a combinatorial optimization problem, where the objective is to find the best way to align units from different models to maximize the effectiveness of the fused model.
### Re-basin
1. permutation symmetry
- The function of the network won't change if the units of hidden layer are exchanged by permutation on some occasion.
- A $\ell$-layer function of DNN $f^{(\ell)}(x,w)=\sigma(W^{\ell-1}f^{\ell-1}+b^{\ell-1})$ is equal to 
$$f^{(\ell)}(x,w)=P^T\sigma(PW^{\ell-1}f^{\ell-1}+Pb^{\ell-1})$$
2. general alignment process
- {{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724852858/image-6_ovbduc.png" thumbnail="" >}}
- adjust the parameter vectors of the two neurons in different hidden layers are close to the replacement point. At the replacement point, the two neurons compute the same function, which means that two neurons can be exchanged(?).
1. re-basin
- Based on permutation symmetry, solutions from diverse area in weight space can generate equivalent solutions. A equivalent solution is located in a same region as the original one with low-loss barrier.   
$$f^{(\ell)}(x,w)=\sigma(P^{(\ell)}W^{(\ell)}(P^{(\ell-1)})f^{(\ell)}+P^{(\ell)}b^{(\ell)})$$
(?)
- With optimal permutation matrix $P^*$, model fusion can be implemented
$$W=\lambda_1W_1^{(\ell)}+\lambda_2 P^{(\ell)}W_2^{(\ell)}(P^{(\ell-1)})^T$$
### Activation Matching
1. Basics
- focus on the matching of activation values
- minimize the cost functions between activations
- the purpose is to calculate $P^*$
1. Mathematic Solution
- can be transformed into assignment problems, like linear assignment and quadratic allocation problems
- Hungarian algorithm or Sinkhorn algorithm
- common lost functions:
  - **Correlation-based Measure**:
  $$
  \mathcal{C}(A_m, A_n)_{cor} = \frac{\mathbb{E} \left[ (A_m - \mathbb{E}[A_m]) (A_n - \mathbb{E}[A_n]) \right]}{\xi_m \xi_n}
  $$

  - **Information-theoretic Measure**:
  $$
  \mathcal{C}(A_m, A_n)_{info} = \sum_{a \in A_m^{(W_1)}} \sum_{b \in A_n^{(W_2)}} p(a, b) \log \left( \frac{p(a, b)}{p(a) p(b)} \right)
  $$

  - **Euclidean Distance Measure**:
  $$ 
  \mathcal{C}(A_m, A_n)_{l2} = \left\| A_m^{(W_1)} - P A_n^{(W_2)} \right\|^2
  $$
  - $A_m$: activation of unit m with standard deviation $\xi$
  - $p(a)$ marginal probability distributions
- one another solution is based on optimal transport(OT)
### Weight Matching
1. Basics
- focus on the matching of activation values, like similar patterns
- minimize the cost functions between weights
- the purpose is to calculate $P^*$
2. Mathematic Solution
### Summary
1. pros:
- make models more similar by adjusting params
- improve information sharing between models
- improve the generalization ability
- improve performance and robustness
2. cons:
- overhead
### Reference
- [Samuel Ainsworth - Git Re-Basin: Merging Models modulo Permutation Symmetries](https://www.youtube.com/watch?v=ffZFrvuxjc8)

## Weight Average
### Overview
1. Definition: 
- Weight Average combines the weights of multiple neural network models to form a single model that potentially offers better performance and generalization. This method is known by different names, such as "vanilla average" or "weight summation"
2. Formula
$$\sum \lambda_iW_i$$
3. Conditions
- The models being combined must share part of the training trajectory or be located in the same "basin" in the loss landscape.
- Models should have similar weights but with certain differences to benefit from WA.
- focus on the fusion of convex combinations of solutions in the same basin, which makes the merged solution closer to the midpoint (optima) of the basin with better
generalization performance
### Weight Average
1. Basics
- From a statistical point of view, weighted average alow for control on individdual model parameters, resulting in a reliable effect on regularization properties and output result
- For the solutions before and after fine-tuning, which are usually within a basin, the linear interpolation works and improves the accuracy of the fused model.
$W=(1-t)\cdot{W_0}+t\cdot{W_{ft}}$
2. Mathematic Solutions
- Fisher merging
- RegMean
3. Ideas focusing on increasing the diversity of models 
- PopulAtion Parameter Averaging(PAPA)
   - start at the same initialization(and improve the cosine similarity between networks)
   - train each models on a slightly different data set
   - average these models every few epochs
- fine-tune the base model for multiple times on different auxiliary **tasks** and **re-fine-tune** these auxiliary weights
-  utilize development data and softmax normalized logarithm with **temperature** to adjust the parameters.
4. Ideas focusing on iterative averaging
- average the weights at different times during the training process of the same or architecturally identical model
  
### SWA
1. Basics
- SWA trains a single model to find smoother solutions than SGD.
2. Algorithm
- At the end of each cycle, the SWA model $W_{SWA}$ is updated by averaging the newly obtained weights over the existing weights
$$W_{SWA} \leftarrow \frac{W_{SWA}\cdot n+W}{n+1}$$
3. Cons
- SWA can only average the points near the local optimal point, and finally get a relatively minimum value
-  the final input sample deviation could be large or insufficient due to some factors (e.g., poor convergence at early stage, large learning rate, fast weight change rate, etc.), resulting in bad overall effect
4. Optimize SWA with different sampling schedules
- SWA-Densely(SWAD)
- Latest weight averaging(LAWA)
- SWA in Low-Precision (SWALP)
- SWA-Gaussian (SWAG)
- Trainable Weight Averaging (TWA)
- Hierarchical Weighted Average (HWA)
- Exponential Moving Average (EMA)
- ...
### Model Soup
1. basics:
- Average the fine-tuned models with different htperparameters
- Reduce the inference time required for ensemble learning $\frac{1}{n}\sum_{i=1}^n f(x,W_i)$
- including uniform soup, greedy soup, adversarially-robust model soup
1. Uniform Soup
- average all the weights of the model directly
- $f(x,\frac{1}{n}\sum_{i=1}^n W_i)$
2. Greedy Soup
-  adds the models to the soup in sequence, keeping the model in the soup if the accuracy of the verification set does not decrease
-  performs the best  
$ingredients \leftarrow ingredients\cup\{W_i\}\text{if Acc}(Avg(ingredients\cup \{W_i\})) \geq Acc(Avg(ingredients))$
- can be viewed as another form of SWA
3. Adversarially-robust model soup
- The adversarially-robust model soup moves the convex hull of parameters of each classifier to adjust the weights of soup, in order to balance the robustness to different threat models and adapt to potential attacks
### Model Arithmetic
- MTL
### Average in Subspace

## Ensemble Learning
### Ensemble Learning
### Model Reuse


