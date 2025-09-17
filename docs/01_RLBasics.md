# 1. RL Basics

## 1.1 What is RL?

Reinforcement Learning (RL) aims to answer the question: **how can an agent learn the best behavior in a complex environment by trying different actions and adjusting based on rewards**? By continuous trial and error, the agent gradually learns a decision-making strategy that maximizes cumulative rewards (returns).

Compared with classical supervised learning:  
- **Supervised learning** learns a mapping from features to labels by fitting the data distribution.  
- **RL**, on the other hand, emphasizes learning from **interaction with the environment**, optimizing the agent’s policy to maximize long-term rewards.  

How does RL help LLMs?  
If we model each **action** as “choosing the next token in a sentence,” then **LLM text generation** becomes a sequential decision problem. Thus, RL can be applied to optimize LLMs.

---

## 1.2 Basic Concepts

Below is a concise table summarizing the core concepts in reinforcement learning:

*(Table content to be inserted here)*

---

## 1.3 Policy Gradient

### 1.3.1 Background

The goal of RL is to solve for an optimal policy $\pi^*(a \mid s)$ that maximizes expected cumulative rewards. This can be framed as an optimization problem, solvable by gradient-based methods.

Suppose the policy $\pi$ is parameterized by $\theta$ (e.g., neural network weights), written as $\pi(a \mid s; \theta)$, abbreviated $\pi_\theta$. The objective is to find optimal parameters $\theta^*$ that maximize:

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\Big[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\Big] 
= \mathbb{E}_{s_0 \sim V^{\pi_\theta}(s_0)}[V^{\pi_\theta}(s_0)].
$$

By computing the gradient $\nabla_\theta J(\theta)$ and updating via gradient ascent:  
$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$,  
we iteratively approach the optimal policy. This is the foundation of **policy gradient methods**.

---

### 1.3.2 Derivation

The basic policy gradient formula can be written as:

$$
\nabla_{\theta} J(\theta) = \sum_{t=0}^{\infty} \Pr^{\pi_{\theta}}(s_0 \to s_t)\sum_{a_t} \pi_{\theta}(a_t \mid s_t) \gamma^t q^{\pi_{\theta}}(s_t, a_t) \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t).
$$

where $\Pr^{\pi_\theta}(s_0 \to s_t)$ is the probability of reaching state $s_t$ from $s_0$ under policy $\pi_\theta$.

Further simplification yields:

$$
\nabla_{\theta} J(\theta) 
= \frac{1}{1 - \gamma} \mathbb{E}_{x \sim D^{\pi_{\theta}}, a \sim \pi_{\theta}}[q^{\pi_{\theta}}(x, a) \nabla_{\theta} \log \pi_{\theta}(a \mid x)].
$$

> **Note:** The advantage function $A(s, a) = Q(s, a) - V(s)$ reduces variance by subtracting a baseline $V(s)$.  
> Temporal-Difference (TD) error provides a practical approximation:  
> $\delta_t = r_t + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$.

---

## 1.3.3 A Concrete Example: REINFORCE

*(Details of REINFORCE implementation can be added here.)*

---

## 1.3.4 Limitations of Policy Gradient

- High variance in gradient estimates → mitigated by baselines or advantage functions.  
- Low sample efficiency → improved by methods like PPO and A3C with constrained updates.  

---

## 1.4 Actor–Critic Algorithms

REINFORCE directly uses observed returns $Q$ for updates, which is inefficient. A natural generalization: train a **Critic** (neural network) to estimate $Q$. The Critic evaluates state–action values, while the Actor updates the policy.

Generalized policy gradient formulation:

$$
\nabla_{\theta} J(\theta) \propto \mathbb{E}_{\pi_\theta}[q^{\pi_\theta}(s, a)\nabla_{\theta} \log \pi_\theta(a \mid s)].
$$

Different instantiations include:
- **REINFORCE:** uses full return $Q$  
- **Q Actor–Critic:** uses estimated $Q(s, a)$  
- **Advantage Actor–Critic (A2C):** uses $A(s, a)$  
- **TD Actor–Critic:** uses TD error $\delta$

> **Note:** Advantage reduces variance by centering around a baseline $V(s)$. TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ avoids direct Q-learning bias.

---

## References

- [David Silver’s RL Course Lecture 7: Policy Gradient (PDF)](https://www.davidsilver.uk/wp-content/uploads/2020/03/pg.pdf)

