# MIT 6.S183 / 6.S184：DDPM、Flow Matching、Score Matching 与 CFG 统一笔记

> 本文整理 MIT 6.S183 与 MIT 6.S184 中关于 DDPM、Flow Matching、Score Matching、Classifier-Free Guidance（CFG）的训练与生成过程，并用 probability path、ODE、SDE 将它们放入同一个框架。
>
> 为兼容 GitHub、Codex 电脑客户端、网页端和手机端，公式以纯文本代码块为主，不依赖特定 LaTeX 渲染器。

## 0. 资料范围与核心结论

主要对照资料：

- [6.S184 完整讲义](<MIT6.184/lecture_notes.pdf>)：第 2-5 节。
- [6.S184 Lecture 2](<MIT6.184/20260122_Lecture_02.pdf>)：Flow Matching。
- [6.S184 Lecture 3](<MIT6.184/20260123_Lecture_03.pdf>)：Score Matching、SDE sampling 与 CFG。
- [6.S184 Lab 1](<MIT6.184/iap-diffusion-labs/solutions/lab_one_complete.ipynb>)：Euler 与 Euler-Maruyama。
- [6.S184 Lab 2](<MIT6.184/iap-diffusion-labs/solutions/lab_two_complete.ipynb>)：Flow Matching 与 Score Matching。
- [6.S184 Lab 3](<MIT6.184/iap-diffusion-labs/solutions/lab_three_complete.ipynb>)：CFG Flow Matching。
- [6.S183 Lecture 1](<MIT6.183/Lecture 1_ Introduction to Diffusion.pdf>)：noise/denoiser 训练与生成。
- [6.S183 Lecture 2](<MIT6.183/Lecture 2_ Perspectives on Diffusion.pdf>)：Probability Flow ODE、reverse SDE、DDIM 与 DDPM。
- [6.S183 Lecture 3](<MIT6.183/Lecture 3_ Conditional Diffusion.pdf>)：Score Matching、Classifier Guidance 与 CFG。
- [6.S183 Pset 1](<MIT6.183/pset1.ipynb>) 与 [Pset 2](<MIT6.183/pset2.ipynb>)：DDPM/DDIM 与 CFG 实现。

先给出最重要的分类：

| 名称 | 它主要是什么 | 网络常见输出 | 常见生成方式 |
|---|---|---|---|
| DDPM | 特定离散 Gaussian diffusion 的训练与随机采样方法 | noise `epsilon_theta` | 离散反向马尔可夫链 |
| Flow Matching | 学习 probability path 对应速度场的方法 | velocity `u_theta` | ODE |
| Score Matching | 学习各时刻分布 score 的方法 | score `s_theta` | ODE 或 SDE |
| CFG | 条件训练与推理期引导机制，不是独立生成模型 | 可作用于 noise、score 或 velocity | 由基础采样器决定 |

核心统一关系：

```text
同一条 Gaussian probability path
            |
            +--> velocity 参数化 --> Flow Matching
            |
            +--> score 参数化 ----> Score Matching
            |
            +--> noise 参数化 ----> DDPM simple loss

训练后的输出可以相互转换，再选择：

ODE 生成  --> 确定性轨迹
SDE 生成  --> 过程中持续注入随机性
DDPM      --> 特定 diffusion 的离散随机反向过程

CFG 位于“模型输出”和“采样器更新”之间。
```

---

# 1. 两套时间方向必须先分清

## 1.1 6.S184 的生成时间方向

6.S184 为了统一 Flow Model 和 Diffusion Model，通常直接按生成方向定义时间：

```text
t=0：初始噪声 p_init
t=1：数据分布 p_data

X_0 ~ p_init
X_1 ~ p_data
```

本文讲 Flow Matching、Score Matching 和 6.S184 的 SDE extension 时使用这个方向。

## 1.2 DDPM 的传统时间方向

DDPM 通常按前向加噪方向编号：

```text
x_0：干净数据
x_T：近似标准高斯噪声
```

训练时可以直接从 `x_0` 构造任意 `x_k`；生成时从 `x_T` 反向迭代：

```text
x_T -> x_(T-1) -> ... -> x_1 -> x_0
```

因此两套记号的端点对应为：

```text
6.S184 的 X_0  <--> DDPM 的 x_T
6.S184 的 X_1  <--> DDPM 的 x_0
```

它们只是时间重编号，不是理论冲突。

## 1.3 6.S183 Pset 与 6.S184 的 Flow Matching 方向

6.S183 Pset 1 的 Flow Matching 练习采用“数据到噪声”的插值：

```text
x_t=(1-t)·z+t·epsilon

t=0：数据 z
t=1：噪声 epsilon

velocity target=epsilon-z
```

生成时从 `t=1` 的噪声出发，向 `t=0` 反向积分。

6.S184 则把生成方向直接记为正时间：

```text
x_tau=tau·z+(1-tau)·epsilon

tau=0：噪声 epsilon
tau=1：数据 z

velocity target=z-epsilon
```

令：

```text
tau=1-t
```

两者就完全一致。velocity target 的符号相反，是因为时间方向相反。

6.S183 Lecture 1 还常使用 additive-noise 记号：

```text
x_sigma=z+sigma·epsilon
```

其中 `sigma` 本身充当噪声坐标。它同样是 Gaussian path，只是没有采用 6.S184 的 `[0,1]` 生成时间参数化。

---

# 2. 统一起点：Gaussian probability path

## 2.1 条件 probability path

设：

```text
z       ~ p_data       # 干净数据
epsilon ~ N(0,I)       # 标准高斯噪声
```

选择两个随时间变化的 schedule：

```text
A(t)：数据系数
B(t)：噪声标准差系数
```

6.S184 讲义通常把它们写作 `alpha_t、beta_t`。本文改写为大写 `A(t)、B(t)`，是为了避免与 DDPM 中的离散 `alpha[k]、beta[k]` 混淆。

定义：

```text
x_t = A(t)·z + B(t)·epsilon
```

等价地：

```text
p_t(x|z)
=
N(
  mean       = A(t)·z,
  covariance = B(t)^2·I
)
```

注意：

```text
B(t)   是标准差系数
B(t)^2 才是方差系数
```

在 6.S184 的生成时间方向下，端点通常满足：

```text
A(0)=0, B(0)=1
A(1)=1, B(1)=0
```

所以：

```text
x_0 = epsilon
x_1 = z
```

## 2.2 边缘 probability path

固定一个 `z` 得到条件路径 `p_t(x|z)`；对所有数据点 `z` 混合，得到真正从噪声分布通向数据分布的边缘路径：

```text
p_t(x)
=
integral p_t(x|z)·p_data(z) dz
```

端点为：

```text
p_0 = p_init
p_1 = p_data
```

## 2.3 “条件 z”与“引导条件 y”不是一回事

本文使用：

```text
z：某一个具体干净数据样本，例如一张训练图片
y：类别或文本 prompt，例如“橘猫”
```

所以：

```text
p_t(x|z)       # 指定一个具体数据终点
u_t(x|z)       # 指向一个具体数据终点的条件速度

u_t(x|y)       # 给定类别或 prompt 后的边缘速度场
s_t(x|y)       # 给定类别或 prompt 后的边缘 score
```

不要把 Flow Matching 中对数据点 `z` 的“conditional”，与 CFG 中对 prompt `y` 的 conditioning 混为一谈。

---

# 3. Gaussian path 下的三个核心目标

## 3.1 条件 score

Gaussian 条件路径为：

```text
p_t(x|z)
=
N(A(t)·z, B(t)^2·I)
```

对 `x` 求 log-density 的梯度：

```text
s_target(x,t|z)
=
gradient_x log p_t(x|z)

=
-[x-A(t)·z] / B(t)^2
```

因为训练样本满足：

```text
x_t-A(t)·z = B(t)·epsilon
```

所以：

```text
s_target(x_t,t|z)
=
-epsilon/B(t)
```

## 3.2 条件 velocity

沿固定的 `(z,epsilon)` 插值轨迹：

```text
x_t=A(t)·z+B(t)·epsilon
```

对时间求导：

```text
v_target
=
dx_t/dt

=
A'(t)·z+B'(t)·epsilon
```

这个目标速度产生所选的条件 probability path。

若要写成 `(x,z,t)` 的函数，先由：

```text
epsilon=[x-A(t)·z]/B(t)
```

得到：

```text
u_target(x,t|z)
=
[A'(t)-B'(t)·A(t)/B(t)]·z
+
[B'(t)/B(t)]·x
```

在训练样本 `x_t=A·z+B·epsilon` 上，这两个写法完全相同：

```text
u_target(x_t,t|z)
=
A'(t)·z+B'(t)·epsilon
```

## 3.3 noise 目标

noise predictor 直接回归构造 `x_t` 时使用的 `epsilon`：

```text
epsilon_target = epsilon
```

对于 Gaussian path，三种目标描述的是同一个带噪后验信息：

```text
score target    = -epsilon/B(t)
noise target    = epsilon
velocity target = A'(t)·z+B'(t)·epsilon
```

---

# 4. 关键推导一：A(t)、B(t) 与 a(t)、b(t)

## 4.1 四个符号的区别

```text
A(t)、B(t)：
人为选择的 probability-path schedule。
它们定义 x_t=A(t)z+B(t)epsilon。

a(t)、b(t)：
由 A、B 及其时间导数推导出的转换系数。
它们把 score 转换成 ODE velocity。
```

因此：

```text
A、B 定义路径。
a、b 转换参数化。
```

## 4.2 从 score 解出 z 和 epsilon

由条件 score：

```text
s_t(x|z)
=
-[x-A(t)·z]/B(t)^2
```

整理得到：

```text
A(t)·z
=
x+B(t)^2·s_t(x|z)
```

因此：

```text
z
=
[x+B(t)^2·s_t(x|z)]/A(t)
```

另一方面：

```text
s_t(x|z)=-epsilon/B(t)
```

所以：

```text
epsilon=-B(t)·s_t(x|z)
```

## 4.3 代入 velocity

条件 velocity 为：

```text
u_t(x|z)
=
A'(t)·z+B'(t)·epsilon
```

代入刚才得到的 `z` 和 `epsilon`：

```text
u_t(x|z)

=
A'(t)/A(t)
·[x+B(t)^2·s_t(x|z)]

-
B'(t)·B(t)·s_t(x|z)
```

展开并收集 `score` 与 `x` 的系数：

```text
u_t(x|z)

=
[
  B(t)^2·A'(t)/A(t)
  -
  B(t)·B'(t)
]
·s_t(x|z)

+
[A'(t)/A(t)]·x
```

定义：

```text
a(t)
=
B(t)^2·A'(t)/A(t)
-
B(t)·B'(t)

b(t)
=
A'(t)/A(t)
```

于是：

```text
u_t(x|z)
=
a(t)·s_t(x|z)+b(t)·x
```

## 4.4 为什么边缘 velocity 也满足同一等式

边缘向量场是条件向量场的后验平均：

```text
u_t(x)
=
E[u_t(x|Z) | X_t=x]
```

代入条件等式：

```text
u_t(x)
=
E[a(t)·s_t(x|Z)+b(t)·x | X_t=x]
```

因为 `a(t)、b(t)、x` 不依赖具体的 `Z`：

```text
u_t(x)
=
a(t)·E[s_t(x|Z) | X_t=x]
+
b(t)·x
```

条件 score 的后验平均就是边缘 score：

```text
E[s_t(x|Z) | X_t=x]
=
s_t(x)
```

所以最终得到：

```text
u_t(x)
=
a(t)·s_t(x)+b(t)·x
```

这就是“只训练 score 网络，也可以恢复 Probability Flow ODE 速度”的依据。

## 4.5 CondOT 直线路径示例

选择：

```text
A(t)=t
B(t)=1-t

A'(t)=1
B'(t)=-1
```

则：

```text
x_t=t·z+(1-t)·epsilon
v_target=z-epsilon
```

转换系数为：

```text
a(t)=(1-t)/t
b(t)=1/t
```

所以：

```text
u_t(x)
=
[(1-t)/t]·s_t(x)
+
[1/t]·x
```

`a(t)、b(t)` 在 `t=0` 形式上会发散，但组合后的真实速度可以存在有限极限。实际实现通常避开精确端点，或直接使用更稳定的 velocity/noise 参数化。

---

# 5. 关键推导二：score prediction 与 noise prediction

## 5.1 定义 noise 参数化

Gaussian path 的条件 score target 是：

```text
s_target(x_t,t|z)
=
-epsilon/B(t)
```

定义网络的 noise 参数化：

```text
epsilon_theta(x_t,t)
=
-B(t)·s_theta(x_t,t)
```

等价地：

```text
s_theta(x_t,t)
=
-epsilon_theta(x_t,t)/B(t)
```

这只是输出参数化转换，并没有引入新的生成假设。

## 5.2 展开 score 误差

```text
s_theta(x_t,t)-s_target(x_t,t|z)

=
-epsilon_theta(x_t,t)/B(t)
-
[-epsilon/B(t)]

=
[epsilon-epsilon_theta(x_t,t)]/B(t)
```

取平方范数：

```text
||s_theta-s_target||^2

=
1/B(t)^2
·
||epsilon_theta-epsilon||^2
```

负号不影响平方范数。

## 5.3 为什么 DDPM simple loss 没有 1/B(t)^2

原始的 weighted score loss 可以写成：

```text
L_score
=
E[
  lambda(t)/B(t)^2
  ·
  ||epsilon_theta(x_t,t)-epsilon||^2
]
```

若选择：

```text
lambda(t)=B(t)^2
```

就得到常见的 noise MSE：

```text
L_noise
=
E[
  ||epsilon_theta(x_t,t)-epsilon||^2
]
```

也可以把它理解成 DDPM simple objective 去掉了原始 score loss 中的 `1/B(t)^2` 权重。

对固定 `t`，乘以正的时间权重不改变最优条件均值；但联合训练所有 `t` 时，它会改变不同噪声等级对优化的相对贡献。因此两种 loss 的最优目标密切相关，但训练动力学和有限容量下的结果不必完全相同。

---

# 6. Flow Matching

## 6.1 它学习什么

Flow Matching 学习边缘 probability path 对应的速度场：

```text
u_theta(x,t)
approx
u_target(x,t)
```

真正的边缘速度涉及整个数据分布，不能直接计算。训练时使用可计算的条件速度：

```text
u_target(x,t|z)
```

网络不接收具体数据终点 `z`，所以 MSE 的最优输出是条件目标的后验平均：

```text
u_theta*(x,t)
=
E[u_target(x,t|Z) | X_t=x]

=
u_target(x,t)
```

条件 FM loss 与理想边缘 FM loss 相差一个不依赖模型参数的条件方差项，因此具有相同的参数梯度和最优解。

## 6.2 Flow Matching 训练伪代码

```text
Algorithm: Gaussian Conditional Flow Matching Training

Require:
    dataset z ~ p_data
    vector-field network u_theta(x,t)
    path schedules A(t), B(t)

1: repeat

2:      Sample clean data:
            z ~ p_data

3:      Sample time:
            t ~ Uniform(0,1)

4:      Sample noise:
            epsilon ~ N(0,I)

5:      Construct path sample:
            x_t
            =
            A(t)·z+B(t)·epsilon

6:      Construct target velocity:
            v_target
            =
            A'(t)·z+B'(t)·epsilon

7:      Predict velocity:
            v_hat=u_theta(x_t,t)

8:      Compute loss:
            L=||v_hat-v_target||^2

9:      Update theta using gradient descent on L

10: until training finishes
```

对于 CondOT 直线路径：

```text
x_t      = t·z+(1-t)·epsilon
v_target = z-epsilon

L
=
||u_theta(x_t,t)-(z-epsilon)||^2
```

## 6.3 Flow Matching 的 ODE 生成伪代码

训练完成后，将网络直接作为 ODE 速度场：

```text
dX_t/dt=u_theta(X_t,t)
```

Euler 采样：

```text
Algorithm: Flow Matching ODE Sampling

Require:
    trained u_theta
    number of steps n

1:  Set:
        t=0
        h=1/n

2:  Sample initial noise:
        X_0 ~ p_init

3:  for i=1,...,n do

4:      Predict velocity:
            v=u_theta(X_t,t)

5:      Euler update:
            X_(t+h)=X_t+h·v

6:      t=t+h

7:  end for

8:  return X_1
```

给定初始噪声 `X_0` 后，ODE 轨迹是确定的。高阶 ODE solver 可以替换 Euler，以降低离散误差。

---

# 7. Score Matching

## 7.1 它学习什么

Score Matching 学习边缘分布的 score：

```text
s_theta(x,t)
approx
gradient_x log p_t(x)
```

边缘 score 不可直接计算，但 Gaussian 条件 score 可计算：

```text
s_target(x_t,t|z)
=
-epsilon/B(t)
```

与 Flow Matching 相同，MSE 回归条件 score 会学到其后验平均：

```text
s_theta*(x,t)
=
E[s_target(x,t|Z) | X_t=x]

=
gradient_x log p_t(x)
```

## 7.2 Denoising Score Matching 训练伪代码

```text
Algorithm: Gaussian Denoising Score Matching Training

Require:
    dataset z ~ p_data
    score network s_theta(x,t)
    path schedules A(t), B(t)
    time weighting lambda(t)

1: repeat

2:      Sample clean data:
            z ~ p_data

3:      Sample time:
            t ~ Uniform(0,1)

4:      Sample noise:
            epsilon ~ N(0,I)

5:      Construct noisy sample:
            x_t=A(t)·z+B(t)·epsilon

6:      Construct conditional score target:
            s_target=-epsilon/B(t)

7:      Predict score:
            s_hat=s_theta(x_t,t)

8:      Compute loss:
            L
            =
            lambda(t)·||s_hat-s_target||^2

9:      Update theta using gradient descent on L

10: until training finishes
```

noise 参数化版本：

```text
epsilon_theta=-B(t)·s_theta

L
=
lambda(t)/B(t)^2
·
||epsilon_theta(x_t,t)-epsilon||^2
```

当 `lambda(t)=B(t)^2` 时，就是常用 noise MSE。

## 7.3 Score Model 的 ODE 生成

Gaussian path 下：

```text
u_theta(x,t)
=
a(t)·s_theta(x,t)+b(t)·x
```

因此 score 网络可以转换成 ODE 速度场：

```text
Algorithm: Score Model ODE Sampling

1:  Sample:
        X_0 ~ p_init

2:  Set:
        t=0
        h=1/n

3:  for i=1,...,n do

4:      Predict score:
            s=s_theta(X_t,t)

5:      Convert to velocity:
            u=a(t)·s+b(t)·X_t

6:      Euler update:
            X_(t+h)=X_t+h·u

7:      t=t+h

8:  end for

9:  return X_1
```

这个过程是确定性的 Probability Flow ODE 风格采样。

## 7.4 Score Model 的 SDE 生成

在 6.S184 的“噪声到数据”时间方向中，若 ODE：

```text
dX_t=u_t(X_t)dt
```

具有边缘分布 `p_t`，则对任意 diffusion coefficient `g(t)>=0`，下面的 SDE 在理想连续极限下具有相同的边缘分布：

```text
dX_t
=
[
  u_t(X_t)
  +
  0.5·g(t)^2·s_t(X_t)
]dt
+
g(t)dW_t
```

Euler-Maruyama 生成伪代码：

```text
Algorithm: Score Model SDE Sampling

Require:
    trained score network s_theta
    diffusion coefficient g(t)
    number of steps n

1:  Set:
        t=0
        h=1/n

2:  Sample:
        X_0 ~ p_init

3:  for i=1,...,n do

4:      Predict score:
            s=s_theta(X_t,t)

5:      Recover probability-flow velocity:
            u=a(t)·s+b(t)·X_t

6:      Construct SDE drift:
            drift
            =
            u+0.5·g(t)^2·s

7:      Sample fresh noise:
            xi ~ N(0,I)

8:      Euler-Maruyama update:
            X_(t+h)
            =
            X_t
            +h·drift
            +g(t)·sqrt(h)·xi

9:      t=t+h

10: end for

11: return X_1
```

`g(t)=0` 时，SDE 退化为 ODE。

---

# 8. DDPM

## 8.1 前向离散加噪过程

DDPM 使用离散时间 `k=1,...,T`，并预先指定：

```text
beta[k] in (0,1)
alpha[k] = 1-beta[k]
alpha_bar[k] = product(alpha[1],...,alpha[k])
alpha_bar[0] = 1
```

一步前向加噪：

```text
q(x_k|x_(k-1))
=
N(
  sqrt(alpha[k])·x_(k-1),
  beta[k]·I
)
```

任意时刻可以直接采样：

```text
x_k
=
sqrt(alpha_bar[k])·x_0
+
sqrt(1-alpha_bar[k])·epsilon

epsilon ~ N(0,I)
```

它正是 Gaussian path 的离散版本：

```text
A[k]=sqrt(alpha_bar[k])
B[k]=sqrt(1-alpha_bar[k])
```

## 8.2 DDPM simple training 伪代码

```text
Algorithm: DDPM Noise-Prediction Training

Require:
    dataset x_0 ~ p_data
    noise predictor epsilon_theta(x_k,k)
    variance schedule beta[1:T]

Precompute:
    alpha[k]=1-beta[k]
    alpha_bar[k]=product(alpha[1],...,alpha[k])

1: repeat

2:      Sample clean data:
            x_0 ~ p_data

3:      Sample timestep:
            k ~ Uniform{1,...,T}

4:      Sample noise:
            epsilon ~ N(0,I)

5:      Construct noisy sample:
            x_k
            =
            sqrt(alpha_bar[k])·x_0
            +
            sqrt(1-alpha_bar[k])·epsilon

6:      Predict noise:
            epsilon_hat
            =
            epsilon_theta(x_k,k)

7:      Compute simple loss:
            L
            =
            ||epsilon_hat-epsilon||^2

8:      Update theta using gradient descent on L

9: until training finishes
```

说明：DDPM 可从 ELBO 推导出按时间加权的训练目标；上面是实践中广泛使用的 simplified noise-prediction objective。它与 Gaussian denoising score matching 只差参数化和时间权重。

## 8.3 DDPM 反向均值和方差

定义真实后验方差：

```text
beta_tilde[k]
=
beta[k]
·
(1-alpha_bar[k-1])/(1-alpha_bar[k])
```

使用 noise predictor 构造反向均值：

```text
mu_theta(x_k,k)

=
1/sqrt(alpha[k])
·
[
  x_k
  -
  beta[k]/sqrt(1-alpha_bar[k])
  ·epsilon_theta(x_k,k)
]
```

标准固定小方差写法中的反向转移是：

```text
p_theta(x_(k-1)|x_k)
=
N(
  mean       = mu_theta(x_k,k),
  covariance = beta_tilde[k]·I
)
```

某些实现固定为其他方差，或让网络学习方差；因此不要把“所有 DDPM 方差都必须等于 `beta_tilde[k]`”当作定义本身。

## 8.4 DDPM ancestral sampling 伪代码

```text
Algorithm: DDPM Ancestral Sampling

Require:
    trained epsilon_theta
    variance schedule beta[1:T]

1:  Sample initial noise:
        x_T ~ N(0,I)

2:  for k=T,T-1,...,1 do

3:      Predict noise:
            epsilon_hat
            =
            epsilon_theta(x_k,k)

4:      Compute reverse mean:
            mu
            =
            1/sqrt(alpha[k])
            ·
            [
              x_k
              -
              beta[k]/sqrt(1-alpha_bar[k])
              ·epsilon_hat
            ]

5:      if k>1 then
            xi ~ N(0,I)
        else
            xi=0
        end if

6:      Sample next state:
            x_(k-1)
            =
            mu+sqrt(beta_tilde[k])·xi

7:  end for

8:  return x_0
```

每一步都包含：

```text
确定性的反向均值
+
新采样的随机高斯增量
```

因此给定同一个 `x_T`，DDPM 生成轨迹仍然是随机的。

## 8.5 DDPM 与 score 的关系

DDPM 的 Gaussian path 中：

```text
B[k]
=
sqrt(1-alpha_bar[k])
```

所以 noise predictor 对应的 score 为：

```text
s_theta(x_k,k)
=
-epsilon_theta(x_k,k)
/
sqrt(1-alpha_bar[k])
```

因此同一个 DDPM noise 网络可以先转换成 score，再用于构造：

```text
随机 reverse-SDE / DDPM 风格采样
或
确定性 Probability Flow ODE / DDIM 风格采样
```

---

# 9. Classifier-Free Guidance

## 9.1 CFG 不是独立模型

CFG 不规定模型必须预测 noise、score 还是 velocity，也不规定使用 ODE 或 SDE。

CFG 只做两件事：

```text
训练时：
随机把真实条件 y 替换为空条件 empty，
让同一个网络同时学会条件输出和无条件输出。

生成时：
放大“条件输出 - 无条件输出”的方向。
```

## 9.2 CFG Flow Matching 训练伪代码

```text
Algorithm: CFG Flow Matching Training

Require:
    paired dataset (z,y) ~ p_data
    conditional vector field u_theta(x,t,y)
    condition-drop probability p_drop

1: repeat

2:      Sample paired data:
            (z,y) ~ p_data

3:      Sample:
            t ~ Uniform(0,1)
            epsilon ~ N(0,I)

4:      Construct path sample:
            x_t=A(t)·z+B(t)·epsilon

5:      Construct velocity target:
            v_target=A'(t)·z+B'(t)·epsilon

6:      With probability p_drop:
            y_used=empty
        Otherwise:
            y_used=y

7:      Predict:
            v_hat=u_theta(x_t,t,y_used)

8:      Compute loss:
            L=||v_hat-v_target||^2

9:      Update theta

10: until training finishes
```

注意：随机丢弃条件不会改变 velocity target；它只改变网络能否看到 `y`。

## 9.3 CFG DDPM / Score 训练

DDPM noise 版本：

```text
x_k
=
sqrt(alpha_bar[k])·x_0
+
sqrt(1-alpha_bar[k])·epsilon

y_used=y 或 empty

L
=
||epsilon_theta(x_k,k,y_used)-epsilon||^2
```

Score 版本：

```text
x_t=A(t)·z+B(t)·epsilon
s_target=-epsilon/B(t)

y_used=y 或 empty

L
=
lambda(t)
·
||s_theta(x_t,t,y_used)-s_target||^2
```

因此 CFG 没有创造新的基础训练标签。

## 9.4 CFG 推理组合

本文采用 6.S183/6.S184 课件和常见库中的约定：

```text
output_uncond
=
model(x,t,empty)

output_cond
=
model(x,t,y)

output_cfg
=
output_uncond
+
w·(output_cond-output_uncond)
```

等价地：

```text
output_cfg
=
(1-w)·output_uncond+w·output_cond
```

在这个约定下：

```text
w=0：无条件输出
w=1：普通条件输出
w>1：沿条件方向外推，强化 prompt
```

有些论文把额外的 guidance strength 记成 `s`，并写成：

```text
output_cfg
=
output_cond+s·(output_cond-output_uncond)
```

此时 `s=0` 才是普通条件输出。比较 guidance scale 时必须先确认所用约定。

## 9.5 CFG + Flow ODE

```text
u_uncond=u_theta(X_t,t,empty)
u_cond  =u_theta(X_t,t,y)

u_cfg
=
u_uncond+w·(u_cond-u_uncond)

X_(t+h)
=
X_t+h·u_cfg
```

## 9.6 CFG + DDPM

```text
epsilon_uncond
=
epsilon_theta(x_k,k,empty)

epsilon_cond
=
epsilon_theta(x_k,k,y)

epsilon_cfg
=
epsilon_uncond
+
w·(epsilon_cond-epsilon_uncond)
```

然后用 `epsilon_cfg` 替代原来的 `epsilon_hat`，计算 DDPM 反向均值并采样：

```text
mu_cfg
=
1/sqrt(alpha[k])
·
[
  x_k
  -
  beta[k]/sqrt(1-alpha_bar[k])
  ·epsilon_cfg
]

x_(k-1)
=
mu_cfg+sqrt(beta_tilde[k])·xi
```

## 9.7 CFG + Score ODE/SDE

```text
s_uncond=s_theta(X_t,t,empty)
s_cond  =s_theta(X_t,t,y)

s_cfg
=
s_uncond+w·(s_cond-s_uncond)
```

随后：

```text
ODE：
用 s_cfg 转换出 velocity，再做 ODE 更新。

SDE：
用 s_cfg 构造 drift，再做 Euler-Maruyama 更新。
```

## 9.8 CFG 的理论边界

`w=1` 对应模型学习的普通条件场。`w>1` 是外推：

```text
output_cfg
!=
真实条件分布对应的原始 output_cond
```

因此 CFG 在 `w>1` 时是经验有效的启发式方法，不再保证精确采样真实的 `p_data(x|y)`。通常条件遵循性提高，但多样性可能下降，过大的 scale 还可能导致过饱和或结构失真。

---

# 10. ODE 与 SDE 的统一描述

## 10.1 ODE：速度场搬运概率质量

ODE：

```text
dX_t=u_t(X_t)dt
```

其边缘密度满足连续性方程：

```text
partial_t p_t(x)
=
-div(p_t(x)·u_t(x))
```

含义是：向量场 `u_t` 搬运概率质量，使分布沿 `p_t` 演化。

Flow Matching 直接学习 `u_t`；Score Matching 可以先学习 `s_t`，再通过：

```text
u_t(x)=a(t)·s_t(x)+b(t)·x
```

恢复 ODE 速度。

## 10.2 SDE：drift 搬运 + diffusion 扩散

一般 SDE：

```text
dX_t=f_t(X_t)dt+g(t)dW_t
```

其密度满足 Fokker-Planck 方程：

```text
partial_t p_t(x)
=
-div(p_t(x)·f_t(x))
+
0.5·g(t)^2·Laplacian(p_t(x))
```

如果已知某条 probability path 的 ODE 速度 `u_t` 与 score `s_t`，选择：

```text
f_t(x)
=
u_t(x)+0.5·g(t)^2·s_t(x)
```

就得到 6.S184 的 SDE extension：

```text
dX_t
=
[
  u_t(X_t)+0.5·g(t)^2·s_t(X_t)
]dt
+
g(t)dW_t
```

它和 ODE 在理想连续极限下具有相同的每时刻边缘分布 `p_t`，但样本路径不同。

## 10.3 Euler-Maruyama 中隐含的均值和方差

SDE 的 Euler-Maruyama 一步为：

```text
X_(t+h)
=
X_t+h·f_t(X_t)+g(t)·sqrt(h)·xi

xi ~ N(0,I)
```

给定 `X_t=x` 后，这一步等价于条件高斯：

```text
X_(t+h) | X_t=x

~
N(
  mean       = x+h·f_t(x),
  covariance = h·g(t)^2·I
)
```

所以连续时间 SDE 课件虽然不显式写“均值、方差”，但离散更新已经隐式定义了：

```text
一步均值 = 当前状态 + drift·步长
一步方差 = diffusion_coefficient^2·步长
```

这与 DDPM 的：

```text
x_(k-1)
=
mu_theta(x_k,k)
+
sqrt(beta_tilde[k])·xi
```

具有相同的“均值 + 高斯随机增量”结构。

## 10.4 传统 diffusion 时间方向下的 reverse SDE

在传统的“数据到噪声”连续时间方向中，设前向 SDE 为：

```text
dX_t
=
f_t(X_t)dt+g(t)dW_t

t: 0 -> T
```

从 `T` 向 `0` 积分的 reverse-time SDE 可写成：

```text
dX_t
=
[
  f_t(X_t)-g(t)^2·s_t(X_t)
]dt
+
g(t)dWbar_t

其中积分方向为 T -> 0，因而 dt<0。
```

对应的 Probability Flow ODE 为：

```text
dX_t
=
[
  f_t(X_t)-0.5·g(t)^2·s_t(X_t)
]dt

同样从 T 积分到 0。
```

这里的符号与 6.S184 的“噪声到数据”SDE extension 看起来不同，是因为时间方向和基准 drift 的定义不同。比较公式前必须先统一时间方向。

## 10.5 DDPM、DDIM 与连续时间视角

在合适的连续时间极限下：

```text
DDPM ancestral sampling
约对应
特定 reverse-time SDE 的离散随机采样
```

而：

```text
DDIM 的确定性极限 / Probability Flow sampler
约对应
Probability Flow ODE 的离散采样
```

这里的“约对应”很重要：标准 DDPM 首先是离散时间马尔可夫链；把它解释成 SDE 离散化需要指定连续时间极限、schedule 和离散约定，不能在任意系数下把两者逐项机械等同。

---

# 11. 四种方法的详细对比

## 11.1 训练过程

| 方法 | 构造带噪样本 | 网络输入 | 网络输出 | 训练目标 |
|---|---|---|---|---|
| DDPM | `sqrt(alpha_bar)·x_0 + sqrt(1-alpha_bar)·epsilon` | `x_k,k` | noise | `epsilon` |
| Flow Matching | `A(t)·z+B(t)·epsilon` | `x_t,t` | velocity | `A'(t)·z+B'(t)·epsilon` |
| Score Matching | `A(t)·z+B(t)·epsilon` | `x_t,t` | score | `-epsilon/B(t)` |
| CFG | 与基础模型相同 | 额外输入 `y` 或 `empty` | noise/score/velocity 任一种 | 与基础模型相同 |

共同点：

```text
1. 从数据集中采样干净样本。
2. 随机采样时间或离散 timestep。
3. 采样 Gaussian noise。
4. 解析构造带噪输入与监督标签。
5. 使用 MSE 回归某种等价参数化。
```

主要区别是网络回归的坐标不同。

## 11.2 生成过程

| 方法 | 初始化 | 单步确定项 | 单步随机项 | 典型轨迹 |
|---|---|---|---|---|
| Flow ODE | `X_0~p_init` | `h·u_theta` | 无 | 确定 |
| Score ODE | `X_0~p_init` | score 转换出的 `h·u_theta` | 无 | 确定 |
| Score SDE | `X_0~p_init` | `h·[u+0.5g^2s]` | `g·sqrt(h)·xi` | 随机 |
| DDPM | `x_T~N(0,I)` | 反向均值 `mu_theta` | `sqrt(beta_tilde)·xi` | 随机 |
| CFG | 与基础采样器相同 | 先组合有/无条件输出 | 由基础采样器决定 | 由基础采样器决定 |

## 11.3 网络输出之间的转换

Gaussian path 下：

```text
score:
s_theta(x,t)
=
-epsilon_theta(x,t)/B(t)
```

```text
noise:
epsilon_theta(x,t)
=
-B(t)·s_theta(x,t)
```

```text
velocity:
u_theta(x,t)
=
a(t)·s_theta(x,t)+b(t)·x
```

因此，在转换非退化且 schedule 已知时：

```text
学习 noise
<-->
学习 score
<-->
学习 velocity
```

理论上描述的是同一 probability path 的不同参数化；实践中它们的数值尺度、损失权重、端点稳定性和优化难度不同。

---

# 12. 完整统一流程

## 12.1 训练阶段

```text
选择 probability path
    |
    |  x_t=A(t)·z+B(t)·epsilon
    |
    +--> 选择 velocity target
    |       |
    |       +--> Flow Matching
    |
    +--> 选择 score target=-epsilon/B(t)
    |       |
    |       +--> Score Matching
    |
    +--> 选择 noise target=epsilon
            |
            +--> DDPM simple objective

如果需要条件生成：

把 y 输入网络，训练时以一定概率替换为 empty
            |
            +--> 获得同一网络的 conditional/unconditional 输出
```

## 12.2 生成阶段

```text
初始化 Gaussian noise
        |
        v
调用训练好的网络
        |
        |-- 如果使用 CFG：
        |      分别计算 conditional/unconditional 输出并组合
        |
        |-- 如果输出是 noise：
        |      可换算 score 或直接计算 DDPM reverse mean
        |
        |-- 如果输出是 score：
        |      可换算 probability-flow velocity 或 SDE drift
        |
        |-- 如果输出是 velocity：
        |      可直接作为 ODE 速度；Gaussian path 下也可恢复 score
        |
        v
选择采样动力系统
        |
        +--> ODE：Euler / Heun / 高阶 solver / DDIM 风格
        |
        +--> SDE：Euler-Maruyama / reverse-SDE 风格
        |
        +--> DDPM：离散 ancestral sampler
        |
        v
得到数据样本
```

## 12.3 最简统一伪代码

```text
Algorithm: Unified Conditional Generative Sampling

1: Initialize noise state x

2: for each sampling time do

3:      If CFG is enabled:
            out_uncond=model(x,time,empty)
            out_cond=model(x,time,y)
            out=out_uncond+w·(out_cond-out_uncond)
        Else:
            out=model(x,time,y or empty)

4:      Convert out when necessary:
            noise <-> score <-> velocity/drift

5:      Apply the selected sampler:

            ODE:
                x_next=x+h·velocity

            SDE:
                x_next=x+h·drift+g·sqrt(h)·xi

            DDPM:
                x_next=mu_theta+sigma_step·xi

6:      x=x_next

7: end for

8: return x
```

---

# 13. 高频易错点

## 13.1 “DDPM 和 Score Matching 是两套互不相关的训练”

不准确。DDPM 常用 noise MSE 是 Gaussian denoising score matching 的 noise 参数化，并改变了不同时间的 loss 权重。

## 13.2 “Flow Matching 只能 ODE，Score Matching 只能 SDE”

不准确。Flow Matching 最自然地直接得到 ODE velocity；Score Matching 最自然地得到 score。但 Gaussian path 下两者可以转换，因此都可以构造 ODE；拥有 score 后也可以构造 SDE。

## 13.3 “CFG 是一种新的 diffusion sampler”

不准确。CFG 先修改模型输出，随后仍然使用原来的 ODE、SDE、DDPM 或其他 sampler。

## 13.4 “条件向量场就是 prompt 条件向量场”

不准确。Flow Matching 推导中的 `u_t(x|z)` 指定具体数据点 `z`；prompt conditioning 使用的是 `u_t(x|y)`。两种条件的作用不同。

## 13.5 “SDE 公式没写均值和方差，所以与 DDPM 不同”

不准确。Euler-Maruyama 更新隐式定义了小时间步的条件高斯均值和方差：

```text
mean       = x+h·drift
covariance = h·g(t)^2·I
```

## 13.6 “去掉 SDE 随机项就一定得到正确 DDIM”

不准确。若要保持同一边缘 probability path，从 SDE 改为 Probability Flow ODE 时，drift 也要从 SDE drift 改成对应的 ODE velocity，不能只把随机项删掉。

## 13.7 “同一 probability path 唯一决定一个向量场”

一般不成立。Probability path 只规定各时刻的边缘分布；可能存在多个动力系统产生相同边缘路径。Flow Matching 选择一个可构造、可监督的条件向量场，并通过边缘化学到相应的边缘场。

## 13.8 “a(t)、b(t) 是另一套 noise schedule”

不准确：

```text
A(t)、B(t)：定义 Gaussian probability path。
a(t)、b(t)：由 A、B 推导，用于 score -> velocity 转换。
```

## 13.9 “所有端点公式都能直接数值计算”

不一定：

```text
A(t)=0 时，A'(t)/A(t) 可能发散。
B(t)=0 时，-epsilon/B(t) 可能发散。
```

理论等式通常在非退化的内部时间成立；实现会避开精确端点，使用截断噪声范围，或选择数值更稳定的参数化和预条件。

---

# 14. 一句话总复习

```text
Probability path 规定每个时刻“分布应该是什么”。

Flow Matching 学习“样本应该以什么速度移动”。

Score Matching 学习“当前位置朝哪个方向概率密度上升最快”。

DDPM 用 noise prediction 表示 score，并通过离散随机反向转移生成。

CFG 同时使用条件和无条件输出，放大 prompt 造成的方向变化。

ODE 与 SDE 决定模型输出如何被积分成完整生成轨迹：
ODE 确定性搬运概率质量，SDE 在 drift 之外持续加入随机扩散。
```
