---
title: "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs - Note"
date: 2024-05-29
tags:
- model fusion
- note
thumbnailImagePosition: left
thumbnailImage: https://res.cloudinary.com/dsssaawdu/image/upload/v1724852873/image-2_wfa7m5.png
---

This post is the note of the paper "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs". See [docs](https://arxiv.org/abs/1802.10026) for more info.
<!--more-->
# Introduction
## loss surfaces
- highly non-convex
- depend on a large amount of parameters
## Isolated
- The number of local optima and saddle points is large
- The loss is high along a line segment connecting two optima
## Visualization
- This is the $l_{2}$-regularized cross-entropy \
$L(y,\hat{y})=-\frac{1}{N}\sum_{i=1}^{N}[y_{i}\log{(\hat{y_{i}})}+(1-y_{i})\log{(1-\hat{y_{i}})}]$\
$R(w)=\lambda \sum_{i=1}^{p}{w_{i}}^{2}$\
$L_{total}(y,\hat{y},w)=L(y,\hat{y})+R(w)$\
train load surface of a ResNet-164 on CIFAR-100 as a function of network weights in a 2-dimensional subspace. It's defined by three independently trained networks. All optima are isolated\
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724852871/image_gv1hwj.png" thumbnail="" >}}
- A quadratic Bezier curve connecting the lower two optima on the panel above along a path of near-constant loss.\
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724852872/image-1_upzdqf.png" thumbnail="" >}}
- A polygonal chain with one bend connecting the lower two optima on the panel above along a path of near-constant loss.\
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724852873/image-2_wfa7m5.png" thumbnail="" >}}
The text in the image describes a method to visualize a loss function in the plane defined by three weight vectors $w_1, w_2, w_3$ in a neural network or a similar model. Here's an explanation of the formulae and steps involved:
- axis:
1. **Defining vectors $u$ and $v$**:
   - $u = w_2 - w_1$
   - $v = w_3 - w_1 - \frac{\langle w_3 - w_1, w_2 - w_1 \rangle}{\| w_2 - w_1 \|^2} \cdot (w_2 - w_1)$
   
   These vectors $u$ and $v$ are defined to span the plane containing $w_1, w_2, w_3$. Vector $u$ is the difference between $w_2$ and $w_1$. Vector $v$ is the projection of $w_3 - w_1$ onto the orthogonal complement of $w_2 - w_1$.

2. **Normalizing vectors $u$ and $v$**:
   - $\hat{u} = \frac{u}{\| u \|}$
   - $\hat{v} = \frac{v}{\| v \|}$
   
   These are the unit vectors in the directions of $u$ and $v$, respectively.

3. **Creating an orthonormal basis**:
   - The vectors $\hat{u}$ and $\hat{v}$ form an orthonormal basis in the plane containing $w_1, w_2, w_3$.

4. **Visualizing the loss in the plane**:
   - A Cartesian grid is defined in the basis $\hat{u}, \hat{v}$.
   - The networks are evaluated corresponding to each point in this grid.
   - A point $P$ with coordinates $(x, y)$ in the plane is given by:
     $$
     P = w_1 + x \cdot \hat{u} + y \cdot \hat{v}
     $$
   
   This means that any point $P$ in the plane can be represented as a combination of the base point $w_1$ and scaled versions of the unit vectors $\hat{u}$ and $\hat{v}$.

# Finding Paths between Modes
## Connection Procedure
- Let $\hat{w_{1}}$ and $\hat{w_{2}}$ in $R^{|net|}$ be two sets of weights corresponding to two neural networks independently trained by minimizing any user-specified loss $\mathcal{L}(w)$. $|net|$ is the number of weights of the DNN. Let $\phi_{\theta} : [0,1]->R^{|net|}$ be a continuous piecewise smooth parametric curve, with parameters $\theta$, such that $\phi_{\theta}(0)=\hat{w_{1}},\phi_{\theta}(2)=\hat{w_{2}}$
To find this path, the paper proposes minimizing the expectation over a uniform distribution on the curve, denoted by $\hat{\ell}(\theta)$.

1. **Expectation over Uniform Distribution on the Curve**:
   $$
   \hat{\ell}(\theta) = \frac{\int \mathcal{L}(\phi_\theta)d\phi_\theta}{\int d\phi_\theta} = \frac{\int_0^1 \mathcal{L}(\phi_{\theta}(t))\|{\phi_{\theta}'(t)\|dt}}{\|{\phi_{\theta}'(t)dt}\|} = \int_0^1 \mathcal{L}(\phi_\theta(t)) q_\theta(t) dt = \mathbb{E}_{t \sim q_\theta(t)}[\mathcal{L}(\phi_\theta(t))]
   $$
   - $\mathcal{L}$ is the loss function, such as cross-entropy loss.
   - $q_\theta(t)$ is a distribution over $t \in [0, 1]$.
   - $\phi_\theta(t)$ represents the weights on the curve parameterized by $t$.

2. **Defining $q_\theta(t)$**:
   $$
   q_\theta(t) = \frac{\|{\phi_\theta}(t)\|}{\int_0^1 \|{\phi_\theta}'(t)\| dt}
   $$
   - ${\phi_\theta}'(t)$ is the derivative of $\phi_\theta(t)$ with respect to $t$.
   - The denominator normalizes the distribution on the curve.

3. **Simplified Loss Function**:
   Since the gradients of $\hat{L}(\theta)$ are intractable, the paper proposes a simpler loss function $\ell(\theta)$:
   $$
   \ell(\theta) = \int_0^1 \mathcal{L}(\phi_\theta(t)) dt = \mathbb{E}_{t \sim U(0,1)}[\mathcal{L}(\phi_\theta(t))]
   $$
   - $U(0, 1)$ is the uniform distribution over the interval \([0, 1]\).

4. **Optimization Procedure**:
   To minimize $\ell(\theta)$, the paper uses the following approach:
   - Sample $\hat{t}$ from $U(0, 1)$.
   - Make a gradient step for $\theta$ with respect to $\mathcal{L}(\phi_\theta(\hat{t}))$:
     $$
     \nabla_\theta \mathcal{L}(\phi_\theta(\hat{t})) \approx \mathbb{E}_{t \sim U(0,1)}\nabla_\theta \mathcal{L}(\phi_\theta(t)) = \nabla_\theta \mathbb{E}_{t \sim U(0,1)}\mathcal{L}(\phi_\theta(t)) = \nabla_\theta \ell(\theta).
     $$
   - This process is repeated until convergence.
## Example
- **Polygonal chain**: The trained networks $\hat{w_1}$ and $\hat{w_2}$ serve as the endpoints of the chain and the bends are the parameters $\theta$ of the curve parametrization.
$$\phi _{\theta}(t) =
\begin{cases}
2(t\theta + (0.5-t)\hat{w_1}) &0\leq t \leq 0.5 \\
2((t-0.5)\hat{w_2}+(1-t)) \theta &0.5 \leq t \leq 1
\end{cases}
$$
- **Bezier curze**: A quadratic Bezier curve $\phi _{\theta}(t)$
with endpoints $\hat{w_1}$ and $\hat{w_2}$ is as follows:
$$\phi _{\theta}(t)=(1-t)^2 \hat{w_1} + 2t(1-t)\theta + t^2\hat{w_2}, 0\leq t \leq 1 $$

# Curve Finding Experiments
## Experiments Result
- The authors train two networks with different random initializations to find two modes. Then tehy use the proposed algorithm above to find a path connecting these two modes
in the weight space with a quadratic Bezier curve and a polygonal chain with one bend. They also connect the two modes with a line segment for comparison. In all experiments tehy optimize the loss
(simplified version), as for Bezier curves the gradient of loss (origin) is intractable, and for polygonal chains it's found that loss (simplified version) to be more stable.\
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724852885/image-3_kosidt.png" thumbnail="" >}}\
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724852885/image-4_tgvoof.png" thumbnail="" >}}\
{{< image classes="fancybox fig-100" src="https://res.cloudinary.com/dsssaawdu/image/upload/v1724852892/image-5_cdcstq.png" thumbnail="" >}}

# Fast Geometric Ensembling
- constructing...