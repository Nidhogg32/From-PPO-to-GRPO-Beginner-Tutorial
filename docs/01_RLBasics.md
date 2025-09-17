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

| Term | Definition / Formula | Key Points |
|---|---|---|
| **Action** | A choice the agent can make at a given time; denote as \(a\) (or \(a_t\)). | Action space can be **discrete** (e.g., up/down/left/right) or **continuous** (e.g., steering angle). |
| **Reward** | Immediate feedback from the environment to the agent’s action; denote as \(r(s,a)\). | The goal is to **maximize long-term cumulative reward**, not just immediate rewards. |
| **State** | A description of the environment at a given time; denote as \(s\). | The state must contain sufficient information to support decision-making (**Markov property**). |
| **Policy** | The rule for selecting an action in state \(s\); denote as \(\pi(a \mid s)\). | Can be **deterministic** \((a=\pi(s))\) or **stochastic** (a probability distribution). |
| **State Value** | Expected return from state \(s\) following policy \(\pi\):  \(\displaystyle V^\pi(s)=\mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty}\gamma^{t} r_t \,\middle|\, s_0=s\right]\). | Reflects long-term value of a state; \(\gamma \in [0,1]\) is the **discount factor**. |
| **Q Function (Action Value)** | Expected return after taking action \(a\) in state \(s\) and then following \(\pi\):  \(\displaystyle Q^\pi(s,a)=\mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty}\gamma^{t} r_t \,\middle|\, s_0=s,\, a_0=a\right]\). | Used to evaluate the **quality of a specific action** (e.g., Q-learning). |
| **Optimal Policy** | Among all policies, the one that maximizes expected cumulative reward; denote as \(\pi^*\). | Satisfies \(\displaystyle \pi^*=\arg\max_{\pi} V^\pi(s)\) for all states \(s\). |
| **Bellman Expectation Equation** | \(\displaystyle V^\pi(s)=\sum_a \pi(a\mid s)\sum_{s'} P(s'\mid s,a)\,[\,r(s,a)+\gamma V^\pi(s')\,]\). | Current state value equals **immediate reward + discounted value** of successor states. |
| **Bellman Optimality Equation** | \(\displaystyle V^*(s)=\max_a \sum_{s'} P(s'\mid s,a)\,[\,r(s,a)+\gamma V^*(s')\,]\). | By recursively maximizing over actions, directly solves the **optimal value function** (e.g., value iteration). |


---

## 1.3 Policy Gradient

### 1.3.1 Background

The goal of RL is to solve for an optimal policy $\pi^*(a \mid s)$ that maximizes expected cumulative rewards. This can be framed as an optimization problem, solvable by gradient-based methods.

Suppose the policy $\pi$ is parameterized by $\theta$ (e.g., neural network weights), written as $\pi(a \mid s; \theta)$, abbreviated $\pi_\theta$. The objective is to find optimal parameters $\theta^*$ that maximize:

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right] = \mathbb{E}_{s_0} \left[ V^{\pi_{\theta}}(s_{0})\right]
$$

By computing the gradient $\nabla_\theta J(\theta)$ and updating via gradient ascent:  
$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$,  
we iteratively approach the optimal policy. This is the foundation of **policy gradient methods**.

---

### 1.3.2 Derivation

The basic policy gradient formula can be written as:

$$
\nabla_{\theta} J(\theta)
= \sum_{t=0}^{\infty} \sum_{s_{t}} \Pr\!\left(s_{0} \to s_{t},\, t,\, \pi_{\theta}\right)
\sum_{a_{t}} \pi_{\theta}\!\left(a_{t} \mid s_{t}\right)
\bigl[\gamma^{t}\, q_{\pi_{\theta}}(s_{t}, a_{t})\, \nabla_{\theta}\log \pi_{\theta}(a_{t} \mid s_{t})\bigr]
$$


where $\Pr^{\pi_\theta}(s_0 \to s_t)$ is the probability of reaching state $s_t$ from $s_0$ under policy $\pi_\theta$.

Further deduction yields:

$$
\begin{aligned}
\nabla_{\theta} J(\theta)
&= \sum_{t=0}^{\infty} \sum_{s_{t}} \Pr\!\left(s_{0} \to s_{t},\, t,\, \pi_{\theta}\right)
   \sum_{a_{t}} \pi_{\theta}\!\left(a_{t} \mid s_{t}\right)
   \Big[\gamma^{t} q_{\pi_{\theta}}\!\left(s_{t}, a_{t}\right)\, \nabla_{\theta}\log \pi_{\theta}\!\left(a_{t} \mid s_{t}\right)\Big] \\
&= \sum_{t=0}^{\infty} \sum_{s_{t}} \gamma^{t} \Pr\!\left(s_{0} \to s_{t},\, t,\, \pi_{\theta}\right)
   \sum_{a_{t}} \pi_{\theta}\!\left(a_{t} \mid s_{t}\right)
   \Big[q_{\pi_{\theta}}\!\left(s_{t}, a_{t}\right)\, \nabla_{\theta}\log \pi_{\theta}\!\left(a_{t} \mid s_{t}\right)\Big] \\
&= \sum_{x \in \mathcal{S}} \sum_{t=0}^{\infty} \gamma^{t} \Pr\!\left(s_{0} \to x,\, t,\, \pi_{\theta}\right)
   \sum_{a} \pi_{\theta}(a \mid x)\Big[q_{\pi_{\theta}}(x, a)\, \nabla_{\theta}\log \pi_{\theta}(a \mid x)\Big] \\
&= \sum_{x \in \mathcal{S}} d^{\pi_{\theta}}(x) \sum_{a} \pi_{\theta}(a \mid x)
   \Big[q_{\pi_{\theta}}(x, a)\, \nabla_{\theta}\log \pi_{\theta}(a \mid x)\Big] \\
&= \frac{1}{1-\gamma} \sum_{x \in \mathcal{S}} D^{\pi_{\theta}}(x) \sum_{a} \pi_{\theta}(a \mid x)
   \Big[q_{\pi_{\theta}}(x, a)\, \nabla_{\theta}\log \pi_{\theta}(a \mid x)\Big] \\
&= \frac{1}{1-\gamma}\,
   \mathbb{E}_{\substack{x \sim D^{\pi_{\theta}} \\ a \sim \pi_{\theta}(\cdot \mid x)}}
   \!\left[q_{\pi_{\theta}}(x, a)\, \nabla_{\theta}\log \pi_{\theta}(a \mid x)\right]
\end{aligned}
$$


> **Note:** The advantage function $A(s, a) = Q(s, a) - V(s)$ reduces variance by subtracting a baseline $V(s)$.  
> Temporal-Difference (TD) error provides a practical approximation:  
> $\delta_t = r_t + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$.

---

## 1.3.3 A Concrete Example: REINFORCE

- The simplest example of a policy gradient algorithm is the Monte Carlo Policy Gradient (REINFORCE), which uses the Monte Carlo method to estimate the Q function:
- 1.Use Current Strategy {\pi_\theta}


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

