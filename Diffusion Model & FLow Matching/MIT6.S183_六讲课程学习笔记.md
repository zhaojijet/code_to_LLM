# MIT 6.S183：A Practical Introduction to Diffusion Models

> 六讲综合学习笔记，根据课程课件与我们的讨论整理。  
> 课程课件使用的正式编号是 **MIT 6.S183 (IAP 2026)**。  
> 本文采用课程中较简洁的 Variance Exploding (VE) 写法：
>
> ```math
> x_t=x_0+\sigma_t\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
> ```

---

## 0. 六讲的整体主线

| Lecture | 核心问题 | 关键结论 |
|---|---|---|
| 1. Introduction | Diffusion 如何训练和生成？ | 给干净数据加不同强度的高斯噪声，训练网络预测噪声或干净样本，再从纯噪声逐步反向更新 |
| 2. Perspectives | DDIM、DDPM、ODE、SDE、score 是什么关系？ | DDIM 是 Probability Flow ODE 的离散化；DDPM 是 Reverse SDE 的离散化；不同参数化可以互相转换 |
| 3. Conditional Diffusion | 怎样按照类别、文本或观测生成？ | 条件 score 可以通过直接条件训练、Classifier Guidance 或 Classifier-Free Guidance 获得 |
| 4. Distillation | 怎样减少采样步数，并把二维扩散先验用于其他任务？ | 可蒸馏多步轨迹为少步模型；也可把预训练 diffusion 当作语义梯度提供者 |
| 5. Applications | 为什么 diffusion 能扩展到图像、视频、机器人？ | Latent diffusion 降低计算量；迭代去噪适合多峰输出、动作序列和数据流形约束 |
| 6. Generalization | 为什么模型没有只记住训练图片？ | 有限训练集的精确经验最优去噪器其实会记忆；真正的泛化来自平滑性、数据结构、SGD 稳定性和早停 |

六讲可以压缩成一条链：

```text
数据分布
  -> 加噪得到一族平滑分布 p_t
  -> 学习 noise / denoiser / score / velocity
  -> 用 ODE 或 SDE 反向穿过这些分布
  -> 用条件或观测改变反向方向
  -> 用蒸馏降低求解成本
  -> 扩展到图像、视频、控制和 3D
  -> 分析模型为什么能生成训练集以外的样本
```

---

# Lecture 1：Introduction to Diffusion Models

## 1.1 为什么需要生成模型

传统监督学习通常学习单值映射：

```math
x\longmapsto y
```

但很多任务是一对多的。例如，同一句“草地上的柯基”对应无数张合理图片。若直接用像素 MSE 做回归，模型倾向于输出所有可能答案的平均，结果往往模糊。

生成模型学习的是分布：

```math
p_\theta(x)
\quad\text{或}\quad
p_\theta(x\mid c)
```

其中条件 `c` 可以是类别、文本、草图、深度图或者其他观测。

## 1.2 VAE 的思路与问题

VAE 引入潜变量 `z`：

```math
q_\phi(z\mid x),\qquad p_\theta(x\mid z)
```

典型目标由两部分组成：

```math
\mathcal L_{\mathrm{VAE}}
=
\mathbb E_{q_\phi(z\mid x)}[-\log p_\theta(x\mid z)]
+
D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\Vert p(z)\right)
```

- 重建项要求 decoder 恢复输入。
- KL 项要求 latent 分布接近简单先验，如标准高斯。

课件强调两个问题：

1. **细节模糊**：当多个样本映射到相似 latent 时，像素回归容易平均不同可能性。
2. **Posterior collapse**：强 decoder 可能忽略 `z`，直接依靠自身能力建模数据分布，使 latent 失去信息。

## 1.3 GAN 的思路与问题

GAN 使用：

- Generator：把随机噪声映射为样本；
- Discriminator：区分真实样本和生成样本。

Generator 通过 Discriminator 的梯度不断改善。但这是 min-max 博弈：

```math
\min_G\max_D\;
\mathbb E_{x\sim p_{\mathrm{data}}}\log D(x)
+
\mathbb E_{z}\log(1-D(G(z)))
```

主要问题：

- Generator 与 Discriminator 的训练速度必须平衡；
- 梯度可能不稳定；
- 容易 mode collapse，只覆盖数据分布的一部分；
- 实际训练常依赖许多技巧。

## 1.4 Diffusion 的核心优势

Diffusion 固定一个明确的加噪过程，再学习逆过程：

```text
数据 x0 -> 逐渐加噪 -> 近似高斯噪声
高斯噪声 -> 模型逐步去噪 -> 数据样本
```

相对 VAE 和 GAN：

- 不需要对抗式 min-max 训练；
- 训练目标通常是稳定的监督 MSE；
- 能较好覆盖多峰分布；
- 模型可以共享于无条件、条件生成、编辑和逆问题；
- 缺点是采样通常需要多次网络调用。

## 1.5 训练数据是怎样构造的

设训练集样本为：

```math
x_0\sim p_{\mathrm{data}}
```

每次训练随机采样：

```math
\sigma\sim p_{\mathrm{train}}(\sigma),
\qquad
\epsilon\sim\mathcal N(0,I)
```

构造带噪样本：

```math
x_\sigma=x_0+\sigma\epsilon
```

把 `(x_\sigma,\sigma)` 输入模型，预测本次加入的噪声：

```math
\epsilon_\theta(x_\sigma,\sigma)
```

训练损失：

```math
\mathcal L(\theta)
=
\mathbb E_{x_0,\sigma,\epsilon}
\left[
\left\|
\epsilon_\theta(x_0+\sigma\epsilon,\sigma)-\epsilon
\right\|^2
\right]
```

课程示例中的 `sigma schedule` 一般覆盖约 `0.01` 到 `100`，常在 `log sigma` 空间设计或采样。

## 1.6 为什么预测随机噪声有意义

单次的 `epsilon` 确实是随机的。同一个 `x_0`、同一个 `sigma`，重新采样就会得到不同的噪声。但是模型看到的不只是噪声，它还看到：

```math
x_\sigma=x_0+\sigma\epsilon
```

对于给定的 `(x_\sigma,\sigma)`，MSE 的最优预测不是复现某一次随机事件，而是条件均值：

```math
\epsilon^*(x_\sigma,\sigma)
=
\mathbb E[\epsilon\mid x_\sigma,\sigma]
```

MSE 可以分解为：

```math
\mathbb E\|\epsilon_\theta-\epsilon\|^2
=
\mathbb E\left\|
\epsilon_\theta-\mathbb E[\epsilon\mid x_\sigma,\sigma]
\right\|^2
+
\mathbb E\operatorname{Var}(\epsilon\mid x_\sigma,\sigma)
```

第二项是不可消除的随机性，模型真正能优化的是第一项。因此网络学习的是：

> 在当前带噪位置，平均而言应该沿哪个方向去掉噪声，才能回到高概率数据区域。

## 1.7 为什么 sigma 不需要训练

`sigma` 在基础 diffusion 中不是模型参数，而是人为定义的：

- 训练时，它决定如何破坏数据；
- 它作为条件输入告诉网络当前噪声强度；
- 推理时，采样器按照预先选定的下降 schedule 使用它。

需要训练的是网络参数 `theta`。当然，也存在学习 noise schedule 的研究，但那是额外设计，不是基础算法的必要组成。

## 1.8 有限训练集上的理想噪声预测器

若训练集是有限集合：

```math
\mathcal K=\{x_0^{(1)},\ldots,x_0^{(N)}\}
```

经验分布下的最优噪声预测器为：

```math
\epsilon^*(x,\sigma)
=
\frac{
\sum_i (x-x_0^{(i)})
\exp\!\left(-\frac{\|x-x_0^{(i)}\|^2}{2\sigma^2}\right)
}{
\sigma
\sum_j
\exp\!\left(-\frac{\|x-x_0^{(j)}\|^2}{2\sigma^2}\right)
}
```

定义后验权重：

```math
w_i(x,\sigma)
=
\frac{
\exp\!\left(-\frac{\|x-x_0^{(i)}\|^2}{2\sigma^2}\right)
}{
\sum_j
\exp\!\left(-\frac{\|x-x_0^{(j)}\|^2}{2\sigma^2}\right)
}
```

则：

```math
\epsilon^*(x,\sigma)
=
\frac{1}{\sigma}\sum_iw_i(x,\sigma)(x-x_0^{(i)})
```

### 分母里的 `j` 到底是什么

分母遍历的是**所有候选干净训练样本** `x_0^(j)`，不是“同一张图片在训练过程中历次抽到的 epsilon”。

直观上，观察到带噪点 `x` 后，我们不知道它来自哪一个干净训练样本。分母负责归一化所有可能来源：

```math
w_i(x,\sigma)
\propto
p(x\mid x_0^{(i)},\sigma)
```

训练时反复为同一图片采样不同 `epsilon`，是在 Monte Carlo 意义上近似整个期望；它不是公式分母索引的含义。

## 1.9 去噪器与平滑距离函数

干净样本预测为：

```math
\hat x_0(x,\sigma)=x-\sigma\epsilon_\theta(x,\sigma)
```

课件把理想去噪方向解释成某个 `sigma` 平滑距离函数的梯度。粗略地说：

- 大 `sigma`：多个训练样本共同影响方向，得到粗粒度全局结构；
- 小 `sigma`：邻近样本主导，恢复高频细节。

## 1.10 DDIM 确定性采样

从大噪声开始：

```math
x_T=\sigma_Tz,\qquad z\sim\mathcal N(0,I)
```

每一步预测当前噪声：

```math
\hat\epsilon_t=\epsilon_\theta(x_t,\sigma_t)
```

课程中的 DDIM 更新为：

```math
x_{t-1}
=
x_t+(\sigma_{t-1}-\sigma_t)
\epsilon_\theta(x_t,\sigma_t)
```

由于：

```math
\sigma_{t-1}<\sigma_t
```

系数为负，因此是在减去一部分模型预测的噪声。

## 1.11 概率采样

课程用一个插值形式说明：更新可以在确定方向之外再加入新高斯噪声：

```math
x_{t-1}
=
x_t+(\sigma_{t'}-\sigma_t)\epsilon_\theta(x_t,\sigma_t)
+\eta w_t,
\qquad
w_t\sim\mathcal N(0,I)
```

这形成从确定性 DDIM 到随机 DDPM 风格采样的一族方法。随机项不是为了“把已经去掉的噪声加回来”，而是为了让采样轨迹具有正确的随机扩散行为，并改善分布探索。

---

# Lecture 2：Perspectives on Diffusion

## 2.1 先区分边缘分布和样本路径

加噪关系：

```math
x_t=x_0+\sigma_t\epsilon
```

定义了每个时间点的边缘分布 `p_t(x)`，但没有唯一决定不同时间点之间如何连接。

可以构造不同路径穿过同一族边缘分布：

- 确定性 ODE 路径；
- 随机 SDE 路径。

它们可以在每个时间点具有相同的 `p_t`，但单个样本的轨迹完全不同。

## 2.2 Probability Flow ODE

设：

```math
\dot x_t=v(x_t,t)
```

密度与速度场满足连续性方程：

```math
\frac{\partial p_t(x)}{\partial t}
+
\nabla\cdot\left(v(x,t)p_t(x)\right)
=0
```

对于课程采用的加噪路径：

```math
x_t=x_0+\sigma_t\epsilon
```

相应速度场为：

```math
v(x_t,t)
=
\dot\sigma_t\,
\mathbb E[\epsilon\mid x_t]
```

训练好的噪声预测器近似该条件期望，所以：

```math
v_\theta(x_t,t)
\approx
\dot\sigma_t\epsilon_\theta(x_t,\sigma_t)
```

## 2.3 Euler 离散化是什么意思

ODE 的积分形式是：

```math
x(t+h)
=
x(t)+\int_t^{t+h}v(x(\tau),\tau)\,d\tau
```

Euler 方法假设短时间内速度基本不变：

```math
x(t+h)
\approx
x(t)+h\,v(x(t),t)
```

反向生成时 `h<0`。把：

```math
v(x_t,t)=\dot\sigma_t\epsilon_\theta(x_t,\sigma_t)
```

代入，利用：

```math
\Delta\sigma\approx\dot\sigma_t h
```

就得到：

```math
x_{t-1}
\approx
x_t+(\sigma_{t-1}-\sigma_t)
\epsilon_\theta(x_t,\sigma_t)
```

因此：

> DDIM 更新不是凭经验发明的减噪公式，而是 Probability Flow ODE 的一阶 Euler 数值积分。

步数越少，单步跨度越大，Euler 近似误差通常越大；也可以用高阶 ODE solver 减少误差。

## 2.4 DDIM 到底怎样跳步

假设训练支持 1000 个噪声等级，但推理只选择：

```text
1000 -> 980 -> 960 -> ... -> 0
```

那么：

- 在 `1000` 处调用一次模型，直接计算 `x_980`；
- 到 `980` 后再次调用模型，计算 `x_960`；
- `999,...,981` 这些状态根本没有被生成。

所以并不是“预测一次噪声，然后把该预测固定用于中间 20 个隐藏步骤”。正确说法是：

> DDIM 把 `1000 -> 980` 当作一个更大的 ODE 数值步，中间 19 个状态和模型调用都被跳过。

经典逐步 DDPM 若走 `1000 -> 999 -> ... -> 980`，需要 20 次反向转移和近似 20 次网络调用。但 DDPM 也存在加速或跳步版本，因此“DDPM 必须永远 1000 步”不是定义本身。

## 2.5 随机微分方程 SDE

一般 SDE：

```math
dx=f(x,t)\,dt+g(t)\,dB_t
```

离散化后：

```math
x_{t+\Delta t}
\approx
x_t+f(x_t,t)\Delta t
+g(t)w_t\sqrt{\Delta t},
\qquad
w_t\sim\mathcal N(0,I)
```

随机项按 `sqrt(Delta t)` 缩放，所以其方差按 `Delta t` 线性增长。

## 2.6 为什么 backward SDE 不只是把 dt 变成负数

ODE 轨迹在足够光滑时可以通过改变时间方向逆转。但 Brownian motion 每一步都会产生新随机信息，不能沿原随机轨迹简单倒放。

因此反向时间 SDE 的 drift 必须加入依赖当前密度的 score 修正。课程采用的反向时间写法对应：

```math
dx
=
f(x,t)\,\overleftarrow{dt}
+
\sqrt{2\rho(t)}\,d\overleftarrow{B}_t
```

其密度演化可写为：

```math
\frac{\partial p_t(x)}{\partial t}
+
\nabla\cdot
\left(
\left[
f(x,t)+\rho(t)\nabla\log p_t(x)
\right]p_t(x)
\right)
=0
```

若希望它与 ODE 的边缘分布相同，需要：

```math
f(x,t)
=
v(x,t)-\rho(t)\nabla\log p_t(x)
```

当：

```math
\rho(t)=0
```

随机项消失，Fokker-Planck 方程退化为 ODE 的连续性方程。

## 2.7 Tweedie 公式

定义 score：

```math
s^*(x_t,t)
=
\nabla_{x_t}\log p_t(x_t)
```

对于高斯加噪：

```math
x_t=x_0+\sigma_t\epsilon
```

Tweedie 公式给出：

```math
\mathbb E[x_0\mid x_t]
=
x_t+\sigma_t^2\nabla\log p_t(x_t)
```

又因为：

```math
\epsilon=\frac{x_t-x_0}{\sigma_t}
```

所以：

```math
\epsilon^*(x_t,t)
=
\mathbb E[\epsilon\mid x_t]
=
\frac{x_t-\mathbb E[x_0\mid x_t]}{\sigma_t}
=
-\sigma_t\nabla\log p_t(x_t)
```

这说明噪声预测并不只是“猜随机噪声”，它与数据密度的 score 完全等价。

## 2.8 四种常见参数化

| 参数化 | 模型目标 | 相互转换 |
|---|---|---|
| Noise prediction | `epsilon*(x_t,t)` | `epsilon=(x_t-d)/sigma_t` |
| Clean sample / denoiser | `d*(x_t,t)=E[x_0\|x_t]` | `d=x_t-sigma_t epsilon` |
| Score prediction | `s*(x_t,t)=nabla log p_t(x_t)` | `s=-epsilon/sigma_t=(d-x_t)/sigma_t^2` |
| Velocity prediction | `v(x_t,t)` | 在本课程路径下 `v=dot(sigma_t) epsilon` |

固定 `sigma_t` 时，它们通过仿射变换等价。但训练时不同参数化会改变：

- 不同噪声等级的损失权重；
- 数值尺度和优化难度；
- 模型对高频、低频信息的关注。

## 2.9 DDIM 与 DDPM 的核心关系

| 项目 | DDIM | DDPM |
|---|---|---|
| 连续对象 | Probability Flow ODE | Reverse SDE |
| 离散方法 | Euler 或其他 ODE solver | Euler-Maruyama 或相应随机离散化 |
| 每步随机性 | `eta=0` 时无新随机噪声 | 通常每步采样新噪声 |
| 给定初始噪声 | 轨迹确定 | 轨迹仍随机 |
| 常见步数 | 容易使用稀疏 schedule | 经典实现多用相邻步，也可加速 |
| 训练模型 | 通常与 DDPM 共用 | 通常与 DDIM 共用 |

“DDIM 用一次 ODE 走 20 步、DDPM 用 20 次 SDE”可以作为数值直觉，但要补充：

- `1000 -> 980` 对 DDIM 是一个大 ODE 步，而不是内部执行 20 个小步；
- DDPM 的 20 次是选择了 20 个相邻随机离散步；
- ODE 与 SDE 在连续极限下可以具有相同边缘分布，但路径不同。

---

# Lecture 3：Conditional Diffusion

## 3.1 Score matching 如何可训练

我们想学习：

```math
s_\theta(x_t,t)
\approx
\nabla_{x_t}\log p_t(x_t)
```

但 `p_t(x_t)` 未知。好在条件加噪分布已知：

```math
p_{t\mid0}(x_t\mid x_0)
=
\mathcal N(x_0,\sigma_t^2I)
```

其 score 是：

```math
\nabla_{x_t}\log p_{t\mid0}(x_t\mid x_0)
=
-\frac{x_t-x_0}{\sigma_t^2}
=
-\frac{\epsilon}{\sigma_t}
```

通过条件期望恒等式：

```math
\mathbb E\left[
\nabla_{x_t}\log p_{t\mid0}(x_t\mid x_0)
\mid x_t
\right]
=
\nabla_{x_t}\log p_t(x_t)
```

所以可训练目标为：

```math
\mathcal L_{\mathrm{SM}}
=
\mathbb E
\left\|
s_\theta(x_t,t)
-
\nabla_{x_t}\log p_{t\mid0}(x_t\mid x_0)
\right\|^2
```

它和噪声预测 MSE 只是尺度不同。

## 3.2 直接训练条件 diffusion

若有带条件的数据：

```math
\mathcal D=\{(x_0^{(i)},c^{(i)})\}_{i=1}^N
```

直接把条件输入模型：

```math
s_\theta(x_t,t,c)
\approx
\nabla_{x_t}\log p_t(x_t\mid c)
```

推理时把条件 score 代入原来的 ODE 或 SDE 即可，采样算法本身没有根本变化。

条件可以是：

- 离散类别；
- 文本 embedding；
- 图像、边缘、深度、姿态；
- 机器人观测或目标；
- 逆问题中的测量 `y`。

## 3.3 Classifier Guidance (CG)

由 Bayes 公式：

```math
p_t(x_t\mid c)
=
\frac{p_t(x_t)p_t(c\mid x_t)}{p(c)}
```

对 `x_t` 求对数梯度：

```math
\nabla_x\log p_t(x_t\mid c)
=
\nabla_x\log p_t(x_t)
+
\nabla_x\log p_t(c\mid x_t)
```

因此需要训练两个模型：

1. 无条件 diffusion score：

```math
s_\theta(x_t,t)\approx\nabla_x\log p_t(x_t)
```

2. 噪声条件分类器：

```math
p_\phi(c\mid x_t,t)
```

分类器的训练目标通常是交叉熵：

```math
\mathcal L_{\mathrm{cls}}
=
-\mathbb E_{x_0,c,t,\epsilon}
\log p_\phi(c\mid x_0+\sigma_t\epsilon,t)
```

它必须看过不同噪声等级的图片，因为生成前期的 `x_t` 几乎是噪声。推理时不是更新分类器，而是对输入求梯度：

```math
\nabla_{x_t}\log p_\phi(c\mid x_t,t)
```

增加 guidance 强度：

```math
\tilde s(x_t,t,c)
=
s_\theta(x_t,t)
+
\gamma\nabla_{x_t}\log p_\phi(c\mid x_t,t),
\qquad
\gamma>1
```

### 为什么 CG 不等于 GAN

两者都有一个“分类器样”的网络，但训练关系不同：

- GAN 的 Discriminator 与 Generator 进行动态对抗，目标是区分真样本与生成样本。
- CG 的 classifier 只判断噪声图像属于哪个语义类别。
- diffusion 不需要击败 classifier；classifier 在推理时提供一个固定的条件梯度。
- 没有 generator-discriminator 的 min-max 博弈。

## 3.4 Classifier-Free Guidance (CFG)

CFG 用同一个条件 diffusion 网络同时学习：

```math
s_\theta(x_t,t,c)
\quad\text{和}\quad
s_\theta(x_t,t,\varnothing)
```

训练时，对一部分样本随机删除条件：

```text
(x_t,t,c)          -> 条件预测
(x_t,t,empty/null) -> 无条件预测
```

两种输入使用同一个噪声预测损失：

```math
\mathcal L_{\mathrm{CFG}}
=
\mathbb E
\left\|
\epsilon_\theta(x_t,t,\tilde c)-\epsilon
\right\|^2
```

其中 `tilde c` 有时为真实条件，有时为 null。

### Conditional Diffusion、CG 与 CFG 的训练输入对比

| 方法 | Diffusion 训练输入 | 额外模型 | 推理时 |
|---|---|---|---|
| 直接 Conditional Diffusion | `(x_t,t,c)`，始终保留条件 | 无 | 一次条件预测 |
| Classifier Guidance | diffusion 使用 `(x_t,t)` | classifier 使用 `(x_t,t,c)` 做分类训练 | diffusion score 加 classifier 输入梯度 |
| Classifier-Free Guidance | 同一网络有时输入 `(x_t,t,c)`，有时输入 `(x_t,t,null)` | 无独立 classifier | 通常分别做 conditional 与 unconditional 两次预测 |

由 Bayes 关系：

```math
\nabla\log p_t(c\mid x_t)
=
\nabla\log p_t(x_t\mid c)
-
\nabla\log p_t(x_t)
```

因此 CFG score：

```math
\tilde s
=
s_{\mathrm{uncond}}
+
\gamma(s_{\mathrm{cond}}-s_{\mathrm{uncond}})
```

等价写成：

```math
\tilde s
=
(1-\gamma)s_{\mathrm{uncond}}
+
\gamma s_{\mathrm{cond}}
```

noise prediction 中同样常写为：

```math
\tilde\epsilon
=
\epsilon_{\mathrm{uncond}}
+
\gamma
\left(
\epsilon_{\mathrm{cond}}
-
\epsilon_{\mathrm{uncond}}
\right)
```

### CFG 能否用狗数据生成猫

不能把 CFG 理解成“自动学会训练数据里不存在的类别”。

- 若从头只用狗图片训练，条件空间里没有猫，模型不会可靠生成猫。
- 若基础模型预训练时见过猫，再用狗数据微调，它可能保留一部分猫的能力，也可能遗忘。
- CFG 的作用是加强模型已经学到的“条件与无条件方向之差”，不是凭空创造未学习概念。

### 为什么仍然使用 CFG

虽然每一步通常要做 conditional 和 unconditional 两次前向，但 CFG 有明显优势：

- 不需要单独训练一个能处理所有噪声等级的 classifier；
- 文本是开放、高维条件，很难为每个文本训练传统分类器；
- 条件模型与无条件模型共享参数；
- 实际上常获得更好的提示词一致性和主观质量。

代价是：

- 计算量通常接近翻倍；
- guidance 太强会降低多样性、过饱和或产生伪影。

CG 更直接，但依赖额外 classifier，且 classifier 梯度可能把样本推向“高分类置信度但不自然”的区域。

## 3.5 Guidance 在做什么

经验上，增大 guidance：

- 把采样推向条件分布的高密度区域；
- 图像通常更清晰、更符合条件；
- 效果类似降低采样温度；
- 但覆盖范围和多样性下降。

这是一种质量与多样性的权衡，而不是无条件的“越大越好”。

## 3.6 Inverse Problems

逆问题希望从观测：

```math
y=\mathcal A(x)+n
```

恢复未知干净样本 `x`。例如：

- 去模糊；
- 超分辨率；
- 图像补全；
- MRI/CT 重建；
- 去噪。

目标是后验分布：

```math
p(x\mid y)
```

对应条件 score：

```math
\nabla_x\log p_t(x_t\mid y)
=
\underbrace{\nabla_x\log p_t(x_t)}_{\text{数据先验}}
+
\underbrace{\nabla_x\log p_t(y\mid x_t)}_{\text{测量匹配}}
```

两种方案：

1. **直接方案**：收集 `(x,y)`，训练条件 diffusion。效果直接，但每种观测模型可能都要重训。
2. **通用方案**：训练一个无条件 diffusion prior，再针对不同测量模型构造 likelihood gradient。更灵活，但 `p_t(y|x_t)` 往往不容易准确计算。

---

# Lecture 4：Distillation

## 4.1 Diffusion distillation 与 LLM distillation 是不是同一个意思

广义含义相同：

> 用一个已经学会复杂行为的 teacher，训练更便宜、更小或接口不同的 student，使 student 逼近 teacher 的分布或行为。

但蒸馏对象不同：

| LLM 蒸馏 | Diffusion 蒸馏 |
|---|---|
| 常蒸馏 logits、token 分布、隐状态或推理轨迹 | 常蒸馏 score、velocity、去噪结果或完整采样轨迹 |
| 重点通常是参数量、吞吐和能力保持 | 重点通常是把几十到上千个去噪步压成 1 到数步 |
| student 仍然自回归生成 token | student 可能改变采样器、时间步和生成动力系统 |

课程进一步把“distill”扩展到：

- 更小模型；
- 更少步模型；
- 不同模态；
- 编辑；
- 异常检测。

## 4.2 为什么原始 diffusion 需要多步

自然图像频谱大致呈幂律结构：

- 低频成分能量大，描述轮廓和大结构；
- 高频成分能量小，描述纹理和细节；
- 白噪声在不同频率上功率较均匀。

加噪时，高频细节首先被淹没。反向去噪则大致按：

```text
全局布局 -> 物体形状 -> 局部结构 -> 高频纹理
```

逐层恢复。因此多步推理天然适合 iterative refinement。

## 4.3 训练无关加速与训练式加速

- **训练无关**：使用 DDIM、高阶 ODE solver、稀疏 schedule。
- **需要额外训练**：Progressive Distillation、Consistency Model、MeanFlow 等。

前者减少数值求解成本，后者试图让模型本身直接完成更大跨度的映射。

## 4.4 Rectified Flow

Rectified Flow 试图把复杂、弯曲的传输轨迹“拉直”。轨迹越接近直线，使用少量 Euler 步的误差越小。

核心直觉：

```text
弯曲轨迹 + 大步长 -> 容易偏离
近似直线 + 大步长 -> 少步也能到达目标
```

它与 flow matching 密切相关，但“轨迹更直”不代表训练时不再需要数据或速度场学习。

## 4.5 Progressive Distillation

典型过程：

1. 训练一个 teacher。
2. 对干净图片加噪得到 `x_t`。
3. teacher 连续执行两个 DDIM 步，得到目标状态。
4. student 学习用一个步完成 teacher 的两个步。

完成一轮后，采样步数约减半：

```text
1024 -> 512 -> 256 -> ... -> 4 -> 2 -> 1
```

这是“逐级压缩轨迹”，而不只是复制 teacher 的单次噪声输出。

## 4.6 Consistency Models

一条 Probability Flow ODE 轨迹上的不同点应对应同一个最终干净样本。Consistency Model 学习映射：

```math
f_\theta(x_t,t)\longmapsto x_0
```

并要求同一轨迹上两个时间点的预测一致：

```math
\mathcal L_{\mathrm{CM}}
=
\left\|
f_\theta(x_t,t)
-
\operatorname{stopgrad}
\left(f_{\bar\theta}(x_s,s)\right)
\right\|_2^2
```

若一致性学得足够好，从高噪声状态可以一次或少数几次映射到数据。

## 4.7 MeanFlow

普通 flow model 学习瞬时速度：

```math
v(z_t,t)
```

MeanFlow 学习时间区间 `[r,t]` 上的平均速度：

```math
u(z_t,r,t)
=
\frac{1}{t-r}\int_r^t v(z_\tau,\tau)\,d\tau
```

平均速度直接描述“大跨度应该怎么走”，更适合单步或少步生成。课件给出的关系包括：

```math
u(z_t,r,t)
=
v(z_t,t)
-
(t-r)\frac{d}{dt}u(z_t,r,t)
```

它和 Progressive Distillation 的共同目标都是学习大步更新，但训练构造和理论视角不同。

## 4.8 Score Distillation Sampling (SDS)

SDS 的目标不是把 diffusion 压成一个更小图像模型，而是：

> 把预训练二维图像 diffusion 当作一个可微的语义先验，用它优化另一个表示，例如 3D NeRF。

设 3D 参数为 `psi`，可微 renderer 为：

```math
x=g(\psi,c)
```

其中 `c` 是相机视角。对渲染图加噪：

```math
x_t=\alpha_tg(\psi,c)+\sigma_t\epsilon
```

预训练 diffusion 根据文本 `y` 预测噪声。SDS 使用近似梯度：

```math
\nabla_\psi\mathcal L_{\mathrm{SDS}}
=
w(t)
\left[
\epsilon_\theta(x_t,t,y)-\epsilon
\right]
\frac{\partial g}{\partial\psi}
```

解释：

- `epsilon_theta-epsilon`：当前渲染结果与文本条件图像分布的偏差方向；
- `partial g/partial psi`：把二维图像梯度链式传回 3D 参数；
- 多视角随机渲染使 3D 模型在不同角度都受到二维语义先验约束。

SDS 的关键 takeaway：

> 预训练 diffusion 不只会生成图像，也可以成为其他可微系统的语义梯度来源。

---

# Lecture 5：Applications of Diffusion

## 5.1 噪声等级与训练重点

不同噪声等级对应不同频率和感知尺度：

- 大噪声阶段决定布局、类别和低频结构；
- 小噪声阶段主要修复高频细节；
- 极小噪声对感知质量影响可能有限。

因此可以通过：

- 改变 noise schedule；
- 改变不同 `sigma` 下的 loss weighting；
- 选择 `x_0`、`epsilon`、`v` 等参数化；

重新分配模型容量。

这些参数化在固定 `sigma` 下数学等价，但跨 `sigma` 训练时不等价，因为它们隐式改变损失权重。

## 5.2 Latent Diffusion

直接在 `512 x 512 x 3` 像素空间去噪非常昂贵。Latent Diffusion 分两阶段：

### 阶段一：训练 VAE

```text
image -> encoder -> latent -> decoder -> reconstructed image
```

课件示例把：

```text
512 x 512 x 3
```

压缩到：

```text
64 x 64 x 4
```

VAE 训练可能组合：

- Pixel regression loss：保证基本重建；
- Perceptual loss：保留人眼重要特征；
- Adversarial loss：提升真实感；
- Bottleneck/KL：限制 latent 容量。

### 阶段二：在 latent 空间训练 diffusion

```text
latent noise -> iterative denoiser -> clean latent -> decoder -> image
```

优势：

- 空间尺寸更小，训练和推理便宜；
- 压缩掉感知上不重要的细节；
- diffusion 重点建模语义和结构。

代价：

- VAE 重建误差形成质量上限；
- latent 仍保留空间结构，未必是最紧凑表示；
- 两阶段训练可能不如端到端目标一致。

## 5.3 Video Diffusion

视频模型常使用 spatial-temporal VAE：

```text
video -> spatiotemporal encoder -> latent video
      -> diffusion/flow model
      -> decoder -> video
```

主要难点：

1. **时间一致性**：人物、物体和背景不能逐帧漂移。
2. **运动压缩**：既要压缩空间细节，也要保留运动规律。
3. **任意长度生成**：固定窗口容易，长视频通常还需要分段或自回归扩展。
4. **计算量**：时间维显著增加 token 数和 attention 成本。

## 5.4 DreamBooth 与图像编辑

DreamBooth 使用少量某个特定主体的图片微调预训练 text-to-image diffusion，使一个稀有标识词与该主体绑定。

它解决的是：

```text
通用类别“狗” -> 特定的一只狗
```

然后可以把主体放进训练集中没有出现的新场景。风险包括过拟合少量图片和丢失原模型多样性。

## 5.5 ControlNet

文本条件控制语义，但难以精确控制空间结构。ControlNet 增加结构条件，如：

- Canny edge；
- depth；
- pose；
- segmentation；
- scribble。

经典设计：

- 冻结预训练 diffusion 主干；
- 复制一条可训练条件分支；
- 用 zero convolution 把条件特征注入主干。

初始化时新增分支接近零影响，因此不会立刻破坏原模型；训练后逐渐学会结构控制。

## 5.6 视频编辑、游戏模拟与 World Models

Diffusion 可以建模：

```math
p(\text{future frames}\mid \text{past frames, actions, conditions})
```

应用包括：

- 视频生成和编辑；
- 交互式游戏画面预测；
- 根据动作预测未来世界状态；
- world model 中的视觉模拟。

这里的模型不仅“画一张图”，而是在学习环境状态随时间和动作变化的条件分布。

## 5.7 Diffusion Policy

机器人策略可以把未来动作序列看成要生成的数据：

```math
p(a_{t:t+H}\mid o_t)
```

其中：

- `o_t` 是图像、机器人状态或任务条件；
- `a_{t:t+H}` 是一段未来动作序列。

训练时，对专家示范动作加噪并学习去噪；推理时从噪声动作序列逐步生成可执行动作。

Diffusion Policy 适合机器人控制的原因：

1. **多峰动作分布**：同一任务可能有多种正确动作，不必用 MSE 平均成一个错误动作。
2. **动作序列整体建模**：可以生成时间上协调的一段动作。
3. **迭代计算**：允许策略在输出空间中反复修正。
4. **噪声训练**：提升对偏离专家轨迹、分布外观测的恢复能力。

课件也强调，效果不应完全归功于“随机生成”。架构改进、噪声增强和多步监督计算都可能是关键。

## 5.8 Large Behavioral Models

Large Behavioral Model 将大规模、多任务机器人示范统一建模，类似视觉语言模型向机器人控制的扩展。它通常结合：

- 多模态观察；
- 任务或语言条件；
- 大规模 imitation learning；
- action chunk 或 diffusion/flow 动作生成。

目标是跨机器人、场景和任务复用行为表示。

## 5.9 Diffusion 的优化解释

令数据集合或数据流形为 `K`，距离函数：

```math
\operatorname{dist}_{\mathcal K}(x)
=
\min_{x_0\in\mathcal K}\|x-x_0\|
```

在投影唯一的区域：

```math
\nabla\frac12
\operatorname{dist}_{\mathcal K}^2(x)
=
x-\operatorname{proj}_{\mathcal K}(x)
```

理想 denoiser 与平滑距离函数的梯度相关。DDIM 更新可被理解成对“到数据流形距离”的近似梯度下降：

```math
x_{t-1}
=
x_t+(\sigma_{t-1}-\sigma_t)
\epsilon_\theta(x_t,\sigma_t)
```

因此 diffusion 的迭代去噪可以解释为：

> 把分布外的初始噪声或观测，逐步拉回模型学习到的数据流形。

在机器人中，这对应 manifold adherence：即使当前观测或初始动作偏离专家数据，迭代修正仍可能把结果拉回合理行为区域。

---

# Lecture 6：Generalization in Diffusion Models

## 6.1 什么叫生成模型泛化

非正式定义：

> 模型能生成既真实、又不是训练样本简单复制的新样本。

形式上，diffusion 定义了模型分布：

```math
p_\theta(x)
```

可以比较：

- 真实或测试分布上的平均 log-likelihood；
- 训练集上的经验平均 log-likelihood。

但课程提醒：

- FID 等感知指标并不完美；
- likelihood 估计可能噪声很大；
- 某个低体积区域可以具有高密度，却只有很小概率质量；
- 因此高 likelihood 不一定代表样本更典型或更符合人类感知。

## 6.2 Diffusion 确实会记忆，也确实会生成新内容

课件引用 Somepalli et al. (2022) 对 Stable Diffusion 的研究案例：至少约 `1.88%` 的样本存在训练数据内容复制，足以引起隐私和版权问题；另一方面，约 `98%` 的生成结果仍是新内容。

因此“生成模型泛化”不是二元状态：

```text
模型可能同时学习通用结构，并记忆少量重复、稀有或高权重样本。
```

## 6.3 Diffusion 泛化悖论

Lecture 1 已经给出有限训练集上的精确最优预测器：

```math
\epsilon^*(x,\sigma)
=
\frac{
\sum_i (x-x_0^{(i)})
\exp\!\left(-\frac{\|x-x_0^{(i)}\|^2}{2\sigma^2}\right)
}{
\sigma
\sum_j
\exp\!\left(-\frac{\|x-x_0^{(j)}\|^2}{2\sigma^2}\right)
}
```

当 `sigma` 很小时，距离最近的训练样本几乎获得全部权重：

```math
\epsilon^*(x,\sigma)
\approx
\frac{x-x_{\mathrm{nearest}}}{\sigma}
```

反向采样就会被拉回某个训练样本。也就是说：

> 对有限经验分布，精确实现经验最优去噪器会复现训练集，而不是创造新样本。

这是 Lecture 6 的核心反直觉结论：

> Diffusion 的泛化发生在神经网络没有精确求出有限训练集的经验最优去噪器时。

这里不是说训练损失越高越好，而是说网络通过结构化、平滑化近似捕捉了数据规律，没有变成精确的训练样本查找器。

## 6.4 与传统监督学习的区别

监督学习给定：

```math
\{(x_i,y_i)\}_{i=1}^N
```

通常有许多函数都能达到零训练误差：

```math
f(x_i)=y_i
```

它们在训练点之外的行为不同。SGD 和网络归纳偏置可能从这些插值解中选择接近真实规律的解。所以：

```text
监督学习中，完美拟合训练集仍然可能泛化。
```

有限数据 diffusion 的经验去噪问题更特殊：它的精确最优场基本被有限原子分布决定，而该场会把采样拉回训练原子。因此简单使用“过参数化后仍能插值”“double descent”或“grokking”并不足以解释 diffusion 的新样本生成。

## 6.5 与 Transformer LLM 泛化的区别

首先要区分：

- **Transformer** 是架构；
- **Diffusion** 是建模目标和生成过程。

DiT 就是使用 Transformer 架构的 diffusion，因此二者不天然对立。

自回归 LLM 学习：

```math
p_\theta(w_t\mid w_{<t})
```

其泛化表现为：

- 对未见上下文预测合理 token；
- 组合训练中学到的语言、知识和任务模式；
- 完成没有逐字出现过的提示和任务。

比较如下：

| 方面 | 传统监督模型 | 自回归 LLM | Diffusion |
|---|---|---|---|
| 学习对象 | `x -> y` 函数 | next-token 条件分布 | 多噪声等级下的 score/denoising field |
| 生成过程 | 通常一次前向 | 从左到右逐 token | 从噪声开始反复更新整个样本 |
| 泛化目标 | 未见输入预测正确 | 未见上下文中语言和能力迁移 | 生成真实且非复制的新样本 |
| 典型记忆 | 记住训练标签或特例 | 复述训练文本 | 复制或近似复制训练图片 |
| 参数共享来源 | 特征与层 | token、位置、attention 规则 | 空间位置、噪声等级、局部和全局视觉结构 |

LLM 也会同时发生模式学习和内容记忆。但 diffusion 的悖论更尖锐：有限训练集对应的精确经验最优去噪场本身就是一个记忆解。

## 6.6 为什么现实中的 diffusion 会泛化

课程归纳了几类原因。

### 1. 平滑性

精确记忆型去噪器在训练样本的 Voronoi 边界附近会突然改变方向。神经网络通常具有低频和平滑偏置，学到的方向不会如此剧烈跳变。

平滑后的模型会产生邻近训练样本的 barycenter 或结构组合，而不是精确落到某一个训练点。

### 2. 数据流形结构

自然图像位于高维像素空间中的低维结构附近。若训练样本足够密集，邻近样本的平滑组合仍可能靠近真实数据流形。

所以模型可以复用：

- 局部纹理；
- 边缘和形状；
- 物体部件；
- 空间关系；
- 语义组合规律。

### 3. 架构归纳偏置

ConvNet 具有：

- locality：输出主要依赖邻近像素；
- equivariance：平移输入会相应平移输出。

但课件也指出，非卷积架构有时同样表现出局部性，而且改变训练数据统计可以让 ConvNet 出现非局部敏感性。因此：

```text
局部性既可能来自架构，也可能从数据统计中涌现。
```

### 4. Geometry-adaptive harmonic bases

ReLU denoiser 在局部是分段线性的，可以把输出表示成少量局部基向量的组合。训练目标鼓励模型用稀疏的、与数据几何相关的基描述输入，而不是逐个保存完整训练样本。

但稀疏表示本身不保证不记忆，关键在于这些基是否捕捉可复用几何结构。

### 5. SGD 稳定性

若替换或扰动少量训练样本不会显著改变模型，则模型较难依赖单个样本，因此更容易泛化。

课程展示的现象是：

- 训练早期，在不相交数据集上训练的模型可能生成非常相似的样本；
- 训练足够久后，两者逐渐分化并记忆各自数据。

这支持：

```text
少量 epoch 的 SGD
-> 算法稳定性
-> 学习共享结构
-> 泛化
```

### 6. Early stopping

训练前期往往先学习跨样本共享的粗结构，后期才更容易拟合单个样本的细节。早停不是单纯“让模型没学会”，而是阻止训练继续逼近记忆型经验最优场。

---

# 7. 六讲中的统一数学视角

## 7.1 一个模型，四种输出

从：

```math
x_t=x_0+\sigma_t\epsilon
```

出发：

```math
\hat x_0=x_t-\sigma_t\hat\epsilon
```

```math
\hat s=-\frac{\hat\epsilon}{\sigma_t}
```

```math
\hat v=\dot\sigma_t\hat\epsilon
```

因此 denoiser、noise、score、velocity 描述的是同一个概率路径的不同坐标。

## 7.2 一个边缘分布族，两类路径

```text
Probability Flow ODE --离散化--> DDIM
          |
          | 通过连续性/Fokker-Planck 匹配相同 p_t
          |
Reverse-time SDE -----离散化--> DDPM
```

ODE 与 SDE 可以共享每个时刻的边缘分布，但：

- ODE 给定初值后是确定轨迹；
- SDE 在每一步加入随机增量；
- “同分布”不等于“同一条样本路径”。

## 7.3 条件生成就是修改 score

```math
\nabla\log p_t(x_t\mid c)
=
\nabla\log p_t(x_t)
+
\nabla\log p_t(c\mid x_t)
```

CG 显式用 classifier 计算第二项；CFG 用 conditional 与 unconditional score 的差近似第二项；逆问题则把第二项换成 measurement likelihood。

## 7.4 Distillation 就是压缩动力系统

多步 diffusion 学到的是一个反向动力系统。蒸馏方法分别压缩：

- 两个 teacher 步到一个 student 步；
- 同一轨迹不同点到共同终点；
- 一段区间的瞬时速度到平均速度；
- 预训练 score 到另一模态的优化梯度。

## 7.5 泛化决定这个动力系统通向哪里

- 精确经验最优场：更接近训练样本查找与复现；
- 平滑、稳定、结构化的近似场：更可能沿数据流形生成新组合；
- 过度训练、数据重复或稀有样本：增加记忆风险。

---

# 8. 高频易错点

## 8.1 “epsilon 是随机的，所以监督标签没有意义”

错误。MSE 学到的是：

```math
\mathbb E[\epsilon\mid x_t,t]
```

不是无条件平均：

```math
\mathbb E[\epsilon]=0
```

条件中的 `x_t` 携带了噪声方向和潜在干净数据的信息。

## 8.2 “sigma 也是变量，所以应该让模型训练 sigma”

错误。随机变量不等于可学习参数。`sigma` 是训练数据构造和采样过程中的外部条件，网络参数 `theta` 才由反向传播更新。

## 8.3 “理想去噪权重分母是在统计同一图片不同 epsilon”

错误。分母对所有候选干净训练样本归一化，表示观察 `x_t` 后每个 `x_0^(j)` 成为来源的相对可能性。

## 8.4 “DDIM 从 1000 跳到 980，只预测一次并在内部复用 20 次”

错误。它只执行一个大步，内部 20 个小状态不存在。下一次模型调用发生在已经得到的 `x_980`。

## 8.5 “DDIM 和 DDPM 必须使用不同训练模型”

通常错误。二者经常共享同一个 noise/score 网络，区别主要在推理时选择 ODE 还是 SDE 离散路径。

## 8.6 “DDIM 就是 Flow Matching”

不完全正确。DDIM 是某个 Probability Flow ODE 的离散采样器；Flow Matching 是直接学习速度场的一类训练方法。两者都可以用 ODE 表达，但训练目标和建模出发点不同。

## 8.7 “Classifier Guidance 像 GAN，所以也是对抗训练”

错误。CG classifier 预测语义类别并在推理时提供输入梯度；它不和 diffusion 模型进行真伪对抗。

## 8.8 “CFG 能从狗数据中自动泛化出猫”

错误。CFG 只能增强模型已经通过数据或预训练学到的条件方向。没有猫相关信息，就没有可靠的猫生成能力。

## 8.9 “CFG 比 CG 总是更省计算”

错误。CFG 通常每个采样步要做条件和无条件两次预测。它的优势是无需单独 noisy classifier、适合开放文本条件且效果稳定，不是单步计算更少。

## 8.10 “Diffusion 蒸馏就是把大 U-Net 换成小 U-Net”

不完整。模型尺寸蒸馏只是其中一种；课程重点还包括把多步采样轨迹压缩成少步或一步。

## 8.11 “训练损失越低，Diffusion 泛化一定越好”

错误。对有限训练集，精确经验最优去噪器会倾向于复现训练样本。训练误差、样本质量、覆盖度和记忆风险必须分别评估。

---

# 9. 核心公式速查

### 前向加噪

```math
x_t=x_0+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
```

### Noise prediction loss

```math
\mathcal L
=
\mathbb E
\left\|
\epsilon_\theta(x_t,t)-\epsilon
\right\|^2
```

### 最优 noise predictor

```math
\epsilon^*(x_t,t)
=
\mathbb E[\epsilon\mid x_t,t]
```

### Denoiser

```math
\hat x_0
=
x_t-\sigma_t\epsilon_\theta(x_t,t)
```

### Tweedie

```math
\mathbb E[x_0\mid x_t]
=
x_t+\sigma_t^2\nabla\log p_t(x_t)
```

### Noise 与 score

```math
\epsilon^*(x_t,t)
=
-\sigma_t\nabla\log p_t(x_t)
```

### Probability Flow velocity

```math
v(x_t,t)
=
\dot\sigma_t\epsilon^*(x_t,t)
```

### DDIM

```math
x_{t-1}
=
x_t+(\sigma_{t-1}-\sigma_t)
\epsilon_\theta(x_t,\sigma_t)
```

### Conditional score

```math
\nabla\log p_t(x_t\mid c)
=
\nabla\log p_t(x_t)
+
\nabla\log p_t(c\mid x_t)
```

### CFG

```math
\tilde\epsilon
=
\epsilon_{\mathrm{uncond}}
+
\gamma
\left(
\epsilon_{\mathrm{cond}}
-
\epsilon_{\mathrm{uncond}}
\right)
```

### Inverse problem posterior score

```math
\nabla\log p_t(x_t\mid y)
=
\nabla\log p_t(x_t)
+
\nabla\log p_t(y\mid x_t)
```

### Consistency loss

```math
\mathcal L_{\mathrm{CM}}
=
\left\|
f_\theta(x_t,t)
-
\operatorname{stopgrad}(f_{\bar\theta}(x_s,s))
\right\|^2
```

### SDS gradient

```math
\nabla_\psi\mathcal L_{\mathrm{SDS}}
=
w(t)
\left[
\epsilon_\theta(x_t,t,y)-\epsilon
\right]
\frac{\partial g}{\partial\psi}
```

---

# 10. 建议的复习顺序

第一轮只掌握下面五个问题：

1. `x_t=x_0+sigma_t epsilon` 如何构造训练数据？
2. 为什么 MSE 预测随机 `epsilon` 能学到条件均值？
3. noise、denoiser、score、velocity 怎样互换？
4. DDIM/ODE 与 DDPM/SDE 的差别是什么？
5. 条件 score 怎样由 CFG 或 Bayes 分解得到？

第二轮再学习：

1. Continuity Equation 与 Fokker-Planck；
2. Tweedie 公式；
3. Progressive Distillation、Consistency、MeanFlow；
4. Latent Diffusion、ControlNet、Diffusion Policy；
5. 有限训练集最优去噪器与泛化悖论。

最终应形成这样的统一理解：

> Diffusion 不是简单地“逐步删除一张图片里的随机噪声”，而是在不同平滑尺度上学习数据分布的几何方向，并用 ODE 或 SDE 数值积分把简单噪声分布运输到数据分布。条件控制改变运输方向，蒸馏压缩运输过程，应用把被运输的对象从图像扩展到视频、3D 和动作，而泛化研究则解释这个动力系统为什么会组合出训练集之外的新样本。

---

# 11. 课件索引

1. [Lecture 1: Introduction to Diffusion](MIT6.183/Lecture%201_%20Introduction%20to%20Diffusion.pdf)
2. [Lecture 2: Perspectives on Diffusion](MIT6.183/Lecture%202_%20Perspectives%20on%20Diffusion.pdf)
3. [Lecture 3: Conditional Diffusion](MIT6.183/Lecture%203_%20Conditional%20Diffusion.pdf)
4. [Lecture 4: Distillation](MIT6.183/Lecture%204_%20Distillation.pdf)
5. [Lecture 5: Applications](MIT6.183/Lecture%205_%20Applications.pdf)
6. [Lecture 6: Generalization in Diffusion Models](MIT6.183/Lecture%206_%20Generalization%20in%20diffusion%20models.pdf)

课程反复使用或提到的代表性方向包括：

- DDPM、score-based generative modeling、DDIM；
- Tweedie 公式、Probability Flow ODE、Reverse SDE；
- Classifier Guidance、Classifier-Free Guidance；
- Progressive Distillation、Consistency Models、MeanFlow；
- DreamFusion/SDS、Latent Diffusion、ControlNet、Diffusion Policy；
- diffusion 的平滑性、稳定性、记忆与泛化。
