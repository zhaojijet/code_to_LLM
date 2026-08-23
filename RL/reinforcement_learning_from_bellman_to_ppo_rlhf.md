# 从贝尔曼方程到 PPO：强化学习价值、策略与 RLHF 的统一推导

本文从马尔可夫决策过程出发，依次建立状态价值、动作价值、贝尔曼方程、MC、TD、策略评估、策略改进、策略梯度、REINFORCE with Baseline、Actor-Critic、GAE 和 PPO 之间的关系，并在最后映射到大语言模型 RLHF。

全文围绕一条主线展开：

```text
环境交互产生轨迹
        ↓
回报 G_t 描述一次实际经历的长期收益
        ↓
V^π / Q^π 描述策略 π 下回报的期望
        ↓
贝尔曼方程描述价值函数必须满足的递归关系
        ↓
DP / MC / TD 用不同方式完成策略评估
        ↓
贪心改进或策略梯度完成策略改进
        ↓
REINFORCE → Actor-Critic → GAE → PPO
        ↓
映射到 LLM 的 token 级 RLHF
```

---

## 1. 基本对象与统一符号

### 1.1 马尔可夫决策过程

强化学习通常建模为马尔可夫决策过程 MDP：

```math
\mathcal M=(\mathcal S,\mathcal A,p,\gamma)
```

- `Sₜ = s`：时刻 `t` 的状态；
- `Aₜ = a`：智能体在状态 `s` 执行的动作；
- `Rₜ₊₁ = r`：执行动作后环境返回的即时奖励；
- `Sₜ₊₁ = s′`：下一状态；
- `p(s′, r | s, a)`：给定当前状态和动作，下一状态与奖励的联合分布；
- `γ ∈ [0,1]`：奖励折扣因子；
- `π(a | s)`：策略在状态 `s` 选择动作 `a` 的概率。

一条轨迹写作：

```text
S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T
```

马尔可夫性表示：在给定当前状态和动作后，下一步结果不再依赖更早的历史。

### 1.2 即时奖励、回报与期望价值

即时奖励 `Rₜ₊₁` 只描述当前动作之后立即得到的奖励。回报 `Gₜ` 则包含从时刻 `t` 开始的全部未来奖励：

```math
G_t
=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}
=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
```

它满足递推关系：

```math
G_t=R_{t+1}+\gamma G_{t+1}
```

需要始终区分：

- `Rₜ₊₁` 是一步即时奖励；
- `Gₜ` 是一条实际轨迹上的完整累计回报；
- `V^π`、`Q^π` 是对随机回报取期望得到的价值函数。

### 1.3 有模型与无模型

“模型”特指环境动力学 `p(s′, r | s, a)`，而不是任意神经网络。

- 有模型（model-based）：智能体知道或学习了环境模型，并用它预测动作后果、搜索或规划；
- 无模型（model-free）：不显式学习和使用环境模型进行规划，直接从真实或模拟环境返回的样本 `(s, a, r, s′)` 学习价值或策略。

无模型不代表没有环境，也不代表执行动作后看不到 `r, s′`。区别是：

```text
有模型：执行前可查询 p(s',r|s,a)，评估假设动作的所有可能后果。
无模型：执行后观察本次真实结果 (r,s')，用样本近似未知期望。
```

DQN 的神经网络近似 `Q(s, a)`，没有预测 `p(s′, r | s, a)`，所以 DQN 仍是无模型方法。

---

## 2. 策略函数、状态价值和动作价值

### 2.1 策略函数

随机策略定义为：

```math
\pi(a\mid s)=\Pr(A_t=a\mid S_t=s)
```

对离散动作有：

```math
\sum_{a\in\mathcal A}\pi(a\mid s)=1
```

确定性策略可写成 `a = μ(s)`。深度强化学习通常用带参数的网络表示策略：

```math
\pi_\theta(a\mid s)
```

### 2.2 状态价值函数

状态价值表示：当前处于状态 `s`，从当前动作开始都按策略 `π` 行动时，能够获得的期望回报。

```math
V^\pi(s)
=\mathbb E_\pi[G_t\mid S_t=s]
```

它对当前策略可能选择的动作以及环境的随机转移都取了期望。

### 2.3 动作价值函数

动作价值表示：当前处于状态 `s`，第一步指定执行动作 `a`，从下一状态开始按策略 `π` 行动时，能够获得的期望回报。

```math
Q^\pi(s,a)
=\mathbb E_\pi[G_t\mid S_t=s,A_t=a]
```

二者的时间边界是：

```text
V^π(s)   ：当前动作开始就由 π 决定。
Q^π(s,a) ：当前动作固定为 a，下一状态开始由 π 决定。
```

### 2.4 `V^π` 与 `Q^π` 的转换

如果已经知道 `Q^π` 和策略 `π`，状态价值是动作价值的策略加权平均：

```math
V^\pi(s)
=\sum_a\pi(a\mid s)Q^\pi(s,a)
```

这一方向不需要环境模型。

如果知道 `V^π`，要计算每个动作的 `Q^π`，还需要知道该动作会产生什么奖励和下一状态：

```math
Q^\pi(s,a)
=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma V^\pi(s')\right]
```

因此：

```text
Q^π → V^π：需要策略 π，对动作求加权平均。
V^π → Q^π：需要环境模型，或通过实际执行动作进行采样估计。
```

仅有一个 `V^π(s)` 不能反推出所有 `Q^π(s,a)`。例如均匀策略下，动作价值 `(8,2)` 和 `(6,4)` 都可以得到状态价值 `5`。`Q → V` 的聚合丢失了动作维度的信息。

---

## 3. 贝尔曼期望方程

贝尔曼方程不是某个特定算法，而是价值函数必须满足的递归关系。它来自回报递推式：

```math
G_t=R_{t+1}+\gamma G_{t+1}
```

### 3.1 状态价值贝尔曼期望方程

从定义开始：

```math
V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
```

代入回报递推式：

```math
V^\pi(s)
=\mathbb E_\pi
\left[R_{t+1}+\gamma G_{t+1}\mid S_t=s\right]
```

在下一状态处，未来回报的期望就是 `V^π(s′)`，于是：

```math
V^\pi(s)
=\sum_a\pi(a\mid s)
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma V^\pi(s')\right]
```

直观形式：

```text
当前状态价值
= 按策略选动作
  × 按环境模型发生转移
  ×（即时奖励 + 折扣后的下一状态价值）的期望
```

### 3.2 动作价值贝尔曼期望方程

第一步动作已经固定为 `a`：

```math
Q^\pi(s,a)
=\sum_{s',r}p(s',r\mid s,a)
\left[
r+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
\right]
```

因为：

```math
V^\pi(s')=\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')
```

所以也可写为：

```math
Q^\pi(s,a)
=\mathbb E
\left[R_{t+1}+\gamma V^\pi(S_{t+1})
\mid S_t=s,A_t=a\right]
```

这里必须使用“即时奖励”，不能把完整回报 `Gₜ` 再与下一状态价值相加，否则会重复计算未来奖励。

### 3.3 有模型计算与无模型采样

贝尔曼关系对有模型和无模型方法都成立，区别只是如何计算其中的条件期望。

有模型时可以显式求和：

```math
Q^\pi(s,a)
=\sum_{s',r}p(s',r\mid s,a)
[r+\gamma V^\pi(s')]
```

无模型时执行一次动作得到样本 `(sₜ, aₜ, rₜ₊₁, sₜ₊₁)`，构造：

```math
\widehat Q_t^{(1)}
=r_{t+1}+\gamma V(s_{t+1})
```

它是当前动作价值 `Q^π(sₜ,aₜ)` 的一步单样本估计，而不是下一步动作价值。

---

## 4. 贝尔曼最优方程

### 4.1 最优价值函数

最优状态价值是在所有策略中能够达到的最大状态价值：

```math
V^*(s)=\max_\pi V^\pi(s)
```

最优动作价值是：当前第一步动作固定为 `a`，之后采用最优策略时的最大回报：

```math
Q^*(s,a)=\max_\pi Q^\pi(s,a)
```

它们之间满足：

```math
V^*(s)=\max_aQ^*(s,a)
```

只有最优动作 `a*` 满足：

```math
Q^*(s,a^*)=V^*(s)
```

非最优动作通常满足 `Q*(s,a) < V*(s)`。所以不能笼统地说“最优动作价值等于最优状态价值”。

### 4.2 最优状态价值贝尔曼方程

```math
V^*(s)
=\max_a
\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma V^*(s')\right]
```

有模型时，价值迭代（Value Iteration）反复应用这个最优算子：

```math
V_{k+1}(s)
\leftarrow
\max_a\sum_{s',r}p(s',r\mid s,a)
[r+\gamma V_k(s')]
```

### 4.3 最优动作价值贝尔曼方程

```math
Q^*(s,a)
=\sum_{s',r}p(s',r\mid s,a)
\left[r+\gamma\max_{a'}Q^*(s',a')\right]
```

有模型时可做 Q-value iteration；无模型时，Q-learning 用一次真实转移近似上述期望。

### 4.4 两个最优方程如何互推

从动作价值方程出发，利用：

```math
\max_{a'}Q^*(s',a')=V^*(s')
```

得到：

```math
Q^*(s,a)
=\sum_{s',r}p(s',r\mid s,a)
[r+\gamma V^*(s')]
```

再对当前动作取最大值即可得到 `V*` 方程。

反方向在已知环境模型时也可以：先用 `V*` 和模型计算每个 `Q*(s,a)`，再利用下一状态的最优状态价值关系：

```math
V^*(s')=\max_{a'}Q^*(s',a')
```

但如果手里只有一张 `V*` 表且没有模型，就不能恢复所有动作的 `Q*`，因为最大值已经丢失了其他动作的信息。

---

## 5. 策略评估、策略改进与广义策略迭代

### 5.1 策略评估

策略评估（policy evaluation）是在当前策略 `π` 固定时，估计：

```math
V^\pi(s)\quad\text{或}\quad Q^\pi(s,a)
```

常见求解方法：

- 已知模型：动态规划，直接计算贝尔曼期望；
- 未知模型：MC 使用完整轨迹；
- 未知模型：TD 使用即时奖励和自举目标。

### 5.2 策略改进

策略改进（policy improvement）根据价值信息构造更好的策略。

如果有 `Q^π`，可以贪心改进：

```math
\pi_{\mathrm{new}}(s)
=\arg\max_aQ^\pi(s,a)
```

训练时常使用 `ε-greedy` 保留探索。

如果只有 `V^π`：

- 有模型时，可以利用模型和 `V^π` 对各动作做一步前瞻；
- 无模型时，`V^π(s)` 本身不能比较当前状态下的动作，需要额外的 Actor 学习策略，或改为学习 `Q^π(s,a)`。

### 5.3 广义策略迭代

策略评估与策略改进不必各自完全收敛。二者可以交替进行：

```text
当前策略 π
   ↓ 产生轨迹
评估 V^π / Q^π
   ↓
根据价值改进策略
   ↓
得到新策略 π'
   ↓
继续采样与评估
```

这种相互推动的过程称为广义策略迭代（Generalized Policy Iteration, GPI）。从 MC Control、SARSA 到 Actor-Critic，都可以用这个视角理解。

---

## 6. 蒙特卡洛方法

### 6.1 MC 策略评估

MC 等一条轨迹结束后计算实际回报：

```math
G_t=R_{t+1}+\gamma R_{t+2}+\cdots
```

状态价值估计：

```math
V^\pi(s)
\approx
\operatorname{avg}\{G_t\mid S_t=s\}
```

增量更新：

```math
V(S_t)
\leftarrow
V(S_t)+\alpha[G_t-V(S_t)]
```

动作价值估计：

```math
Q^\pi(s,a)
\approx
\operatorname{avg}\{G_t\mid S_t=s,A_t=a\}
```

```math
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)+\alpha[G_t-Q(S_t,A_t)]
```

MC 不需要环境模型，也不使用已有价值估计构造目标，因此不自举（no bootstrapping）。

### 6.2 MC Control

MC 不仅能评估固定策略，也能做控制：

1. 用当前策略采集完整轨迹；
2. 用回报 `Gₜ` 更新 `Q(s,a)`；
3. 用 `ε-greedy(Q)` 改进策略；
4. 重复上述过程。

### 6.3 MC 的特点

- 优点：目标直接来自实际完整回报，不依赖下一状态价值估计；
- 缺点：通常要等回合结束，回报方差较大；
- 适合：自然分回合、能够获得完整轨迹的任务；
- 在策略梯度中：REINFORCE 也使用 MC 回报，但更新对象是策略而非必须显式更新 `V/Q`。

---

## 7. 时序差分学习：TD Target 与 TD Error

### 7.1 TD 的一般结构

TD 的核心是使用一个包含当前估计的自举目标：

```math
\text{TD error}
=\text{TD target}-\text{current estimate}
```

TD error 不是只属于动作价值。学习 `V` 和学习 `Q` 都可以构造相应的 TD error。

### 7.2 状态价值 TD(0)

TD target：

```math
y_t^{V}
=R_{t+1}+\gamma V(S_{t+1})
```

TD error：

```math
\delta_t^{V}
=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
```

更新：

```math
V(S_t)
\leftarrow
V(S_t)+\alpha\delta_t^{V}
```

`R + γV(S′)` 同时有两种解释：

- 它是当前状态价值 `V(Sₜ)` 的 TD target；
- 在当前动作已经观测到的条件下，它是当前动作价值 `Q(Sₜ,Aₜ)` 的一步样本估计。

### 7.3 SARSA：on-policy TD 动作价值控制

```math
y_t^{\mathrm{SARSA}}
=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})
```

```math
\delta_t^{\mathrm{SARSA}}
=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
```

```math
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)+\alpha\delta_t^{\mathrm{SARSA}}
```

下一动作 `Aₜ₊₁` 是当前行为策略实际选择的动作，因此 SARSA 是 on-policy。

### 7.4 Q-learning：off-policy TD 动作价值控制

```math
y_t^{Q}
=R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')
```

```math
\delta_t^{Q}
=R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)
```

```math
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)+\alpha\delta_t^{Q}
```

行为策略可以带探索，但目标使用贪心动作，因此 Q-learning 是 off-policy，并直接逼近 `Q*`。DQN 只是用神经网络 `Qθ(s,a)` 替代表格，并引入经验回放和目标网络等稳定化技术。

### 7.5 MC 与 TD 的关系

| 方法 | 学习目标 | 是否自举 | 更新时机 | 典型偏差/方差 |
|---|---|---:|---|---|
| MC | 完整回报 `Gₜ` | 否 | 通常回合结束 | 低自举偏差、高采样方差 |
| 一步 TD | `R + γV(S′)` 或相应 Q target | 是 | 每一步 | 较高自举偏差、较低方差 |
| n-step TD | 前 n 步真实奖励 + 末端 bootstrap | 是 | 获得 n 步后 | 介于二者之间 |

在随机环境中，即使 `V = V^π` 完全正确，单条样本的 TD error 也不必为零。正确的固定点条件是：

```math
\mathbb E[\delta_t\mid S_t=s]=0
```

---

## 8. 策略梯度

价值控制通过 `arg maxₐ Q(s,a)` 间接得到策略；策略梯度则直接优化参数化策略 `πθ(a | s)`。

### 8.1 优化目标

定义期望回报目标：

```math
J(\theta)
=\mathbb E_{\tau\sim\pi_\theta}[G(\tau)]
```

轨迹概率为：

```math
p_\theta(\tau)
=p(S_0)
\prod_t
\pi_\theta(A_t\mid S_t)
p(S_{t+1},R_{t+1}\mid S_t,A_t)
```

环境转移项不含策略参数 `θ`。使用 log-derivative trick：

```math
\nabla_\theta p_\theta(\tau)
=p_\theta(\tau)\nabla_\theta\log p_\theta(\tau)
```

如果目标严格定义为从初始状态开始的折扣回报 `J(θ) = E[G₀]`，利用因果性后可得采样形式：

```math
\nabla_\theta J(\theta)
=\mathbb E_{\tau\sim\pi_\theta}
\left[
\sum_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\gamma^tG_t
\right]
```

也可以把 `γᵗ` 吸收到折扣状态访问分布中。定义：

```math
d_\gamma^\pi(s)
=(1-\gamma)\sum_{t=0}^{\infty}
\gamma^t\Pr(S_t=s\mid\pi)
```

则策略梯度定理可写成省略常数比例的形式：

```math
\nabla_\theta J(\theta)
\propto
\mathbb E_{
s\sim d_\gamma^\pi,\,
a\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)
Q^\pi(s,a)
\right]
```

许多 PPO 推导和实现按 rollout 时间步求平均，因此公式中不再显式写外层 `γᵗ`；折扣已经体现在回报、TD target、GAE 以及采用的状态访问加权约定中。阅读不同资料时应先确认这一约定，不能把是否显式出现 `γᵗ` 当成算法本质差异。

### 8.2 因果性与 reward-to-go

时刻 `t` 的动作不能影响它发生之前的奖励。因此更新 `Aₜ` 时，只保留从 `t` 之后开始的回报 `Gₜ`，而不使用整条轨迹中早于 `Aₜ` 的奖励。这称为 causality trick。

### 8.3 Baseline 为什么不改变期望梯度

可以减去任何不依赖当前动作的状态 baseline `b(Sₜ)`：

```math
\mathbb E_{A_t\sim\pi_\theta}
\left[
\nabla_\theta\log\pi_\theta(A_t\mid S_t)b(S_t)
\right]=0
```

因为：

```math
\sum_a\pi_\theta(a\mid s)
\nabla_\theta\log\pi_\theta(a\mid s)
=\nabla_\theta\sum_a\pi_\theta(a\mid s)=0
```

所以 baseline 不改变梯度期望，但可以显著降低方差。

### 8.4 优势函数

最自然的 baseline 是状态价值：

```math
A^\pi(s,a)
=Q^\pi(s,a)-V^\pi(s)
```

它衡量某动作相对当前状态下策略平均表现的好坏：

- `A^π(s,a) > 0`：该动作优于平均水平；
- `A^π(s,a) < 0`：该动作劣于平均水平；
- `A^π(s,a) = 0`：与平均水平相当。

以下各节采用折扣状态访问分布的写法，因此省略显式的外层 `γᵗ`。策略梯度可以写为：

```math
\nabla_\theta J(\theta)
=\mathbb E
\left[
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
A^\pi(S_t,A_t)
\right]
```

---

## 9. REINFORCE with Baseline

REINFORCE 是 MC 策略梯度：使用完整轨迹回报直接更新策略。

### 9.1 MC 优势估计

完整回报是动作价值的单样本估计：

```math
\mathbb E[G_t\mid S_t=s,A_t=a]=Q^\pi(s,a)
```

因此：

```math
\widehat A_t^{\mathrm{MC}}
=G_t-V_\phi(S_t)
```

是优势函数的 MC 估计。

### 9.2 Actor 更新

```math
\theta
\leftarrow
\theta+alpha_\theta
\widehat A_t^{\mathrm{MC}}
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
```

### 9.3 Baseline/Critic 更新

用 MC 回报拟合状态价值：

```math
L_V(\phi)
=\frac12\left[G_t-V_\phi(S_t)\right]^2
```

REINFORCE with Baseline 已经具有 Actor 与价值 baseline 两部分，但它通常等到完整回报可用后再更新，仍具有较高方差。

---

## 10. Actor-Critic

Actor-Critic 将策略学习和价值学习明确拆成两部分：

- Actor：`πθ(a | s)`，负责选择动作；
- Critic：`Vφ(s)` 或 `Qφ(s,a)`，负责评价策略表现。

### 10.1 一步动作价值与优势估计

对于使用状态价值的 Critic：

```math
Q^\pi(s_t,a_t)
=\mathbb E
[R_{t+1}+\gamma V^\pi(S_{t+1})\mid s_t,a_t]
```

无模型 Actor-Critic 用本次真实转移构造一步估计：

```math
\widehat Q_t^{(1)}
=r_{t+1}+\gamma V_\phi(s_{t+1})
```

这估计的是当前 `Q^π(sₜ,aₜ)`，不是下一步 `Q(sₜ₊₁,aₜ₊₁)`。

减去当前状态价值得到：

```math
\widehat A_t^{(1)}
=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)
=\delta_t
```

所以状态价值 TD error 同时可以作为一步优势估计。

### 10.2 Actor 更新

```math
L_{\mathrm{actor}}(\theta)
=-\mathbb E
\left[
\log\pi_\theta(A_t\mid S_t)\,
\operatorname{stopgrad}(\widehat A_t)
\right]
```

最小化该 loss 等价于沿策略梯度方向最大化期望回报。

### 10.3 Critic 更新

一步 TD target：

```math
y_t=r_{t+1}+\gamma V_{\bar\phi}(s_{t+1})
```

Critic loss：

```math
L_V(\phi)
=\frac12
\left[
\operatorname{stopgrad}(y_t)-V_\phi(s_t)
\right]^2
```

`φ̄` 表示构造固定 target 时使用的旧参数或停止梯度的价值预测。

### 10.4 从一步到多步

n-step 目标是：

```math
y_t^{(n)}
=\sum_{k=0}^{n-1}\gamma^kr_{t+k+1}
+\gamma^nV(s_{t+n})
```

对应优势：

```math
\widehat A_t^{(n)}=y_t^{(n)}-V(s_t)
```

一步 TD 方差低但更依赖 Critic；长步目标使用更多真实奖励、偏差通常降低，但方差增大。GAE 的目标就是平滑地混合不同步数。

---

## 11. 广义优势估计 GAE

### 11.1 从 TD error 开始

定义状态价值的一步 TD error：

```math
\delta_t
=r_{t+1}+\gamma V(s_{t+1})-V(s_t)
```

一步优势为：

```math
\widehat A_t^{(1)}=\delta_t
```

二步优势可以展开为：

```math
\widehat A_t^{(2)}
=\delta_t+\gamma\delta_{t+1}
```

一般的 k-step 优势为：

```math
\widehat A_t^{(k)}
=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}
```

### 11.2 为什么混合多个步数

- 小 `k`：更早 bootstrap，依赖 Critic 较多，通常偏差较高、方差较低；
- 大 `k`：使用更多实际奖励，通常偏差较低、方差较高；
- 单一 `k` 很难对所有状态和训练阶段都最佳。

GAE 用 `λ` 对不同步数进行几何加权，从而连续控制偏差—方差权衡。

对于无限时域、`0 ≤ λ < 1`：

```math
\widehat A_t^{\mathrm{GAE}}
=(1-\lambda)
\sum_{k=1}^{\infty}
\lambda^{k-1}\widehat A_t^{(k)}
```

其中归一化权重为：

```math
w_k=(1-\lambda)\lambda^{k-1},
\qquad
\sum_{k=1}^{\infty}w_k=1
```

`λ` 是衰减/混合参数，`1 − λ` 是使几何权重和为 1 的归一化因子。

对于从 `t` 到终点只剩 `K` 步的有限轨迹，更精确的归一化形式为：

```math
\widehat A_t^{\mathrm{GAE}}
=(1-\lambda)
\sum_{k=1}^{K-1}\lambda^{k-1}\widehat A_t^{(k)}
+\lambda^{K-1}\widehat A_t^{(K)}
```

最后一项吸收剩余权重，因此 `λ = 1` 时自然退化为最长步/MC 优势。

### 11.3 化简为 TD-error 累加

将各个 k-step 优势展开并交换求和顺序，可得：

对于固定的 `δₜ₊ₗ`，它会出现在所有 `k ≥ l + 1` 的 k-step 优势中，因此总系数为：

```math
(1-\lambda)\gamma^l
\sum_{k=l+1}^{\infty}\lambda^{k-1}
=(1-\lambda)\gamma^l
\frac{\lambda^l}{1-\lambda}
=(\gamma\lambda)^l
```

所以：

```math
\widehat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{K-1}
(\gamma\lambda)^l\delta_{t+l}
```

无限时域写作：

```math
\widehat A_t^{\mathrm{GAE}}
=\sum_{l=0}^{\infty}
(\gamma\lambda)^l\delta_{t+l}
```

它可以高效地逆序递推：

```math
\widehat A_t
=\delta_t+\gamma\lambda(1-d_t)\widehat A_{t+1}
```

其中 `dₜ` 表示该转移后是否真正终止。终止状态不能继续 bootstrap。

### 11.4 `γ` 与 `λ` 的区别

- `γ`：定义任务目标中未来奖励的时间折扣；
- `λ`：控制优势估计向未来融合多少 TD 信息；
- `γλ`：决定未来 TD error 对当前优势的总衰减速度。

极端情况：

```text
λ = 0：A_hat_t = δ_t，退化为一步 TD 优势。
λ → 1：更多依赖长轨迹；在正确处理终点时趋近 MC 优势 G_t - V(s_t)。
```

### 11.5 GAE 如何构造 Critic 标签

如果 GAE 使用旧 Critic `V_old` 计算，则：

```math
V_t^{\mathrm{target}}
=V_{\mathrm{old}}(s_t)+\widehat A_t^{\mathrm{GAE}}
```

这个目标也称为 `λ-return`。一步时：

```math
V_{\mathrm{old}}(s_t)+\delta_t
=r_{t+1}+\gamma V_{\mathrm{old}}(s_{t+1})
```

正好还原一步 TD target。

不能只让新 Critic 拟合旧 Critic：

```math
[V_\phi(s_t)-V_{\mathrm{old}}(s_t)]^2
```

这个目标只会复制旧预测，不包含轨迹中的新奖励信息。

---

## 12. PPO：受约束的 Actor-Critic 策略更新

普通策略梯度如果单次更新太大，策略可能突然恶化。PPO 的核心是在复用旧策略数据进行多轮优化时，限制新旧策略的概率比变化。

### 12.1 为什么需要重要性采样比率

轨迹由旧策略 `π_old` 采集，但训练时更新的是新策略 `πθ`。定义：

```math
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
```

未裁剪代理目标：

```math
L^{\mathrm{PG}}(\theta)
=\mathbb E_t[r_t(\theta)\widehat A_t]
```

对它求导，并把旧策略概率和优势视为固定量：

```math
\nabla_\theta L^{\mathrm{PG}}
=\mathbb E_t
\left[
\frac{\nabla_\theta\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}
\widehat A_t
\right]
```

等价地：

```math
\nabla_\theta r_t(\theta)
=r_t(\theta)
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
```

所以 loss 公式中没有显式写 `∇θ πθ`，不代表导数消失；反向传播会自动对标量代理目标求导。

### 12.2 PPO-Clip Actor 目标

```math
L^{\mathrm{CLIP}}(\theta)
=\mathbb E_t
\left[
\min\left(
r_t(\theta)\widehat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)
\widehat A_t
\right)
\right]
```

优化器通常最小化 loss，因此 Actor loss 写成：

```math
\mathcal L_{\mathrm{actor}}
=-L^{\mathrm{CLIP}}
```

优势的作用：

- `Âₜ > 0`：提高本次动作概率；
- `Âₜ < 0`：降低本次动作概率；
- `|Âₜ|` 越大：未被裁剪时，更新信号通常越强；
- `Âₜ ≈ 0`：几乎不推动策略更新。

“优势越小，降幅越大”不准确；应该是“优势越负，降低概率的动力越强”。PPO clip 会在概率比越过边界后截断继续获益的方向，从而限制策略更新。

### 12.3 Critic loss

使用旧策略 rollout 计算一次固定的 GAE 和 value target：

```math
V_t^{\mathrm{target}}
=\operatorname{stopgrad}
\left[V_{\mathrm{old}}(s_t)+\widehat A_t^{\mathrm{GAE}}\right]
```

基本 Critic loss：

```math
\mathcal L_V(\phi)
=\frac12\mathbb E_t
\left[
V_\phi(s_t)-V_t^{\mathrm{target}}
\right]^2
```

它训练的是策略对应的状态价值函数，而不是策略函数本身。一些 PPO 实现还会对新旧 value 的变化做可选 clipping，但仍然必须与包含奖励信息的 value target 比较；旧 value 本身不是监督标签。

### 12.4 Entropy loss

离散策略的熵：

```math
\mathcal H(\pi_\theta(\cdot\mid s))
=-\sum_a\pi_\theta(a\mid s)
\log\pi_\theta(a\mid s)
```

为了在最小化 loss 时鼓励高熵，定义：

```math
\mathcal L_{\mathrm{entropy}}
=-\mathbb E_t
[\mathcal H(\pi_\theta(\cdot\mid s_t))]
```

它是软正则项：鼓励探索、避免策略过早变得过于确定，但不保证熵一定高于某个阈值。作用强度由熵系数控制。

### 12.5 PPO 总损失

一种常见的最小化形式为：

```math
\mathcal L_{\mathrm{PPO}}
=\mathcal L_{\mathrm{actor}}
+c_V\mathcal L_V
+c_H\mathcal L_{\mathrm{entropy}}
```

等价写法也常把熵奖励直接以负号加入。阅读实现时必须先确认每一项究竟是在最大化 objective，还是在最小化 loss。

三项分别负责：

| 项 | 更新对象 | 作用 |
|---|---|---|
| Actor loss | 策略 `πθ` | 根据优势提高或降低动作概率，并通过 clip 限制变化 |
| Critic loss | 价值 `Vφ` | 拟合 `V_old + Â_GAE` 构造的 `λ-return` |
| Entropy loss | 策略 `πθ` | 鼓励分布保持一定随机性与探索 |

### 12.6 PPO 完整训练流程

```text
1. 冻结旧策略 π_old 和旧 Critic V_old。
2. 用 π_old 与环境交互，收集 rollout：s_t,a_t,r_{t+1},done_t。
3. 用 V_old 计算每一步 δ_t。
4. 从后向前递推计算 GAE 优势 A_hat_t。
5. 构造固定 Critic 标签：V_target = V_old + A_hat。
6. 可选：对一个 batch 中的 A_hat 做标准化。
7. 将同一批 rollout 切成 minibatch，训练若干 epoch：
   - 计算新旧动作概率比 r_t(θ)；
   - 计算 clipped Actor loss；
   - 计算 Critic value loss；
   - 计算 entropy loss；
   - 反向传播并更新 Actor/Critic。
8. 将更新后的网络设为新的 old 网络，重新采样。
```

优势、旧 log-prob 和 value target 在同一轮 PPO 的多 epoch 更新中通常保持固定并停止梯度。PPO 仍属于近似 on-policy：旧 rollout 只在有限次更新内复用，然后必须用新策略重新采样。

---

## 13. 从贝尔曼方程到 PPO 的完整演进链

### 13.1 第一阶段：定义长期价值

```text
轨迹奖励 → 回报 G_t
回报的条件期望 → V^π(s)、Q^π(s,a)
```

### 13.2 第二阶段：建立递归关系

```text
G_t = R_{t+1} + γG_{t+1}
        ↓ 取条件期望
贝尔曼期望方程
        ↓ 对策略/动作取最优
贝尔曼最优方程
```

### 13.3 第三阶段：求解价值

```text
有环境模型：DP / Value Iteration / Q-value Iteration
无环境模型：
  MC → 使用完整回报
  TD → 使用一步或多步 bootstrap
  SARSA → on-policy Q 控制
  Q-learning → off-policy 逼近 Q*
```

### 13.4 第四阶段：从价值控制到直接优化策略

```text
Q-based control：学习 Q，再通过 argmax / ε-greedy 得到策略
Policy Gradient：直接参数化并优化 π_θ(a|s)
```

### 13.5 第五阶段：降低策略梯度方差

```text
REINFORCE：完整回报 G_t 更新策略
       ↓ 减去状态价值 baseline
REINFORCE with Baseline：G_t - V(s_t)
       ↓ 用 TD bootstrap 代替完整 MC 回报
Actor-Critic：δ_t 作为一步优势
       ↓ 混合不同步数
GAE：Σ(γλ)^l δ_{t+l}
```

### 13.6 第六阶段：限制策略更新幅度

```text
Actor-Critic + GAE
       ↓ 使用旧策略 rollout，多轮复用数据
新旧策略概率比 r_t(θ)
       ↓ clip 限制过大变化
PPO-Clip
```

这条链中最重要的统一认识是：

```text
贝尔曼方程：价值应该满足什么关系。
MC / TD：不知道完整期望时，如何从样本学习价值。
策略评估：当前策略有多好。
策略改进：如何让策略变好。
策略梯度：直接对策略参数做改进。
Critic / GAE：为策略梯度提供低方差、可学习的优势信号。
PPO：在 Actor-Critic + GAE 上限制每轮策略变化，提高稳定性。
```

---

## 14. PPO 到大模型 RLHF 的映射

在经典 token 级语言模型 RLHF 中，可以将自回归生成过程视为一个有限时域 MDP。

这里采用 token 级 MDP 视角；有些资料也会把整条回答视为一个宏动作，但 PPO/GAE 的逐 token 实现通常使用前者。在有限生成序列中常见 `γ = 1` 或非常接近 1，但应以具体实现为准。

| 强化学习概念 | LLM RLHF 中的对应对象 |
|---|---|
| 状态 `sₜ` | prompt 加上已经生成的 token 前缀 |
| 动作 `aₜ` | 下一 token |
| 策略 `πθ(aₜ ∣ sₜ)` | 待训练语言模型的 next-token 分布 |
| 轨迹 `τ` | 一条完整生成回答 |
| 环境 | 自回归上下文更新以及外部奖励计算流程 |
| 奖励 | 奖励模型分数、规则奖励、可验证奖励及 KL 惩罚等 |
| Critic `Vφ(sₜ)` | value head 对当前前缀未来总奖励的预测 |
| 优势 `Âₜ` | 该 token 相对当前前缀平均表现的估计 |
| 旧策略 `π_old` | 采样 rollout 时冻结的策略快照 |
| 参考策略 `π_ref` | 通常为冻结的 SFT/reference 模型，用来限制策略漂移 |

### 14.1 RLHF 中的奖励结构

序列质量奖励往往主要在回答结束时给出。为了约束策略不要偏离参考模型，可以加入逐 token KL 惩罚。一种概念化写法是：

```math
r_t^{\mathrm{total}}
=r_t^{\mathrm{task}}
-\beta
\left[
\log\pi_{\mathrm{rollout}}(a_t\mid s_t)
-\log\pi_{\mathrm{ref}}(a_t\mid s_t)
\right]
```

这里 `π_rollout` 是生成该回答时的策略，通常就是这一轮冻结的 `π_old`；由它计算出的 KL shaping reward 会随 rollout 一起固定。实际实现可能使用不同的 KL 估计和奖励分配方式，阅读具体代码时应以实现为准。

### 14.2 Value head 与 GAE

Value head 预测：

```math
V_\phi(s_t)
\approx
\mathbb E[\text{从当前 token 前缀开始的未来总奖励}]
```

根据 token 级奖励计算：

```math
\delta_t
=r_{t+1}+\gamma V_{\mathrm{old}}(s_{t+1})
-V_{\mathrm{old}}(s_t)
```

再逆序计算：

```math
\widehat A_t
=\delta_t+\gamma\lambda(1-d_t)\widehat A_{t+1}
```

`Âₜ` 用于更新语言模型策略，`V_old(sₜ) + Âₜ` 用于训练 value head。

### 14.3 PPO 在 RLHF 中的作用

PPO Actor 目标提高正优势 token 的相对概率、降低负优势 token 的相对概率，同时通过 `π_new / π_old` 的 clipping 限制同一批 rollout 上的更新幅度。

需要区分两个约束：

```text
PPO clip：限制 π_new 相对 rollout 策略 π_old 的单轮更新幅度。
Reference KL：限制训练策略长期偏离冻结参考策略 π_ref。
```

二者不是同一个对象，也不能相互替代。

一些 LLM RLHF 实现会弱化或省略显式 entropy bonus，因为 reference KL 已经提供了另一种分布正则；是否保留熵项应以具体训练目标和实现为准。

### 14.4 为什么理解经典 RL 对 RLHF 很重要

- 奖励模型只提供轨迹质量信号，Critic 负责把未来奖励传播到各 token 状态；
- GAE 决定 token 优势的偏差—方差权衡；
- PPO ratio 解决旧策略采样、新策略训练之间的分布变化；
- clip、KL 与 entropy 分别处理不同层面的稳定性和探索问题；
- value loss 训练的是回报预测器，不是语言模型 next-token 策略本身，尽管二者可能共享骨干网络。

---

## 15. 常见混淆澄清

### 15.1 “有神经网络就是有模型”

错误。只有预测并用于规划的环境动力学才是 model-based 中的 model。`V/Q/π` 网络不是环境模型。

### 15.2 “无模型不能学习状态价值”

错误。MC、TD 和 Actor-Critic 都能从样本学习 `V^π(s)`。只是单独的 `V(s)` 在没有环境模型或 Actor 时不能直接比较动作。

### 15.3 “TD error 就是两个价值函数相减”

不完整。TD error 是：

```math
\text{即时奖励}+\text{折扣后的下一价值}-\text{当前价值}
```

### 15.4 “R + γV(S') 是下一步动作价值”

错误。给定本次 `(Sₜ,Aₜ,Rₜ₊₁,Sₜ₊₁)`，它估计的是当前动作价值 `Q(Sₜ,Aₜ)`，同时是当前状态价值 `V(Sₜ)` 的 TD target。

### 15.5 “Actor-Critic 只能用一步估计”

错误。它可以使用一步 TD、n-step return、完整 MC 或 GAE。

### 15.6 “GAE 中 λ 是归一化系数”

不准确。`λ` 控制长步估计的几何衰减；`1 − λ` 才是无限几何混合中的归一化因子。

### 15.7 “TD error 为 0 就是新旧网络参数相同”

错误。TD error 描述样本上的贝尔曼残差；即使价值函数正确，随机环境中的单样本 TD error 也可能不为零。新旧网络在采样状态上的输出相同，也不代表参数完全相同。

### 15.8 “PPO Actor loss 中导数消失了”

错误。公式写的是未求导的标量代理 loss，自动微分会在反向传播时产生 `∇θ πθ`。

### 15.9 “Critic 直接拟合 V_old 就可以”

错误。那只会复制旧预测。`V_old + Â_GAE` 才加入新 rollout 的奖励信息并形成新的 value target。

---

## 16. 公式速查

### 16.1 价值与贝尔曼方程

```math
\begin{aligned}
G_t
&=R_{t+1}+\gamma G_{t+1},\\[4pt]
V^\pi(s)
&=\mathbb E_\pi[G_t\mid S_t=s],\\
Q^\pi(s,a)
&=\mathbb E_\pi[G_t\mid S_t=s,A_t=a],\\[4pt]
V^\pi(s)
&=\sum_a\pi(a\mid s)Q^\pi(s,a),\\
Q^\pi(s,a)
&=\mathbb E[R_{t+1}+\gamma V^\pi(S_{t+1})\mid s,a],\\[4pt]
V^*(s)
&=\max_aQ^*(s,a),\\
Q^*(s,a)
&=\mathbb E
\left[
R_{t+1}+\gamma\max_{a'}Q^*(S_{t+1},a')
\mid s,a
\right].
\end{aligned}
```

### 16.2 MC 与 TD

```math
\begin{aligned}
y_t^{\mathrm{MC}}
&=G_t,\\
y_t^{\mathrm{TD}(0),V}
&=R_{t+1}+\gamma V(S_{t+1}),\\
\delta_t^V
&=y_t^{\mathrm{TD}(0),V}-V(S_t)\\
&=R_{t+1}+\gamma V(S_{t+1})-V(S_t),\\
y_t^{\mathrm{SARSA}}
&=R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),\\
y_t^{\mathrm{Q\text{-}learning}}
&=R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a').
\end{aligned}
```

### 16.3 策略梯度、优势与 Actor-Critic

```math
\begin{aligned}
A^\pi(s,a)
&=Q^\pi(s,a)-V^\pi(s),\\
\widehat A_t^{\mathrm{REINFORCE}}
&=G_t-V(S_t),\\
\widehat A_t^{\mathrm{AC}(1)}
&=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
=\delta_t.
\end{aligned}
```

```math
\nabla_\theta J(\theta)
\propto
\mathbb E
\left[
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\widehat A_t
\right]
```

### 16.4 GAE 与 PPO

```math
\begin{aligned}
\widehat A_t^{\mathrm{GAE}}
&=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},\\
\widehat A_t^{\mathrm{GAE}}
&=\delta_t
+\gamma\lambda(1-d_t)\widehat A_{t+1}^{\mathrm{GAE}},\\
V_t^{\mathrm{target}}
&=V_{\mathrm{old}}(S_t)+\widehat A_t^{\mathrm{GAE}},\\
r_t(\theta)
&=\frac{\pi_\theta(A_t\mid S_t)}
{\pi_{\mathrm{old}}(A_t\mid S_t)},\\
L^{\mathrm{CLIP}}(\theta)
&=\mathbb E_t
\left[
\min\left(
r_t(\theta)\widehat A_t,
\operatorname{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
\end{aligned}
```

---

## 17. 最终统一理解

可以用三个相互独立的维度组织全文：

| 维度 | 问题 | 典型答案 |
|---|---|---|
| 学什么 | 学状态、动作还是策略？ | `V(s)`、`Q(s,a)`、`π(a ∣ s)` |
| 怎么学 | 如何获得监督信号？ | 贝尔曼备份、MC、TD、n-step、GAE |
| 如何控制 | 如何让行为变好？ | `arg maxₐ Q(s,a)`、ε-greedy、策略梯度、PPO |

再加上第四个正交维度：

| 是否使用环境模型 | 方法 |
|---|---|
| 使用 `p(s′, r ∣ s, a)` 预测与规划 | model-based |
| 直接从交互样本学习价值或策略 | model-free |

PPO 的本质可以压缩成一句话：

> PPO 是一种无模型、近似 on-policy 的 Actor-Critic 方法：Critic 基于贝尔曼关系和采样轨迹计算 GAE 与 value target，Actor 使用 GAE 作为策略梯度的优势信号，并通过新旧策略概率比的 clipping 限制每轮策略更新幅度。

理解这句话，就把贝尔曼方程、MC/TD、策略评估、优势函数、Actor-Critic、GAE 与 PPO 串成了一条完整链路。
