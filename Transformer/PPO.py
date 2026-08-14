"""
PPO (Proximal Policy Optimization) 算法实现 — Acrobot-v1 环境
==============================================================

PPO 是一种 On-Policy 的策略梯度方法，通过 **截断 (clip)** 策略比率来限制
每次更新的步幅，从而在不使用 KL 散度约束的情况下实现稳定训练。

核心思想：
  1. 用当前策略 π_θ 采集一批轨迹数据
  2. 计算 GAE (Generalized Advantage Estimation) 得到优势估计 A_t
  3. 用 PPO-Clip 目标函数多轮 (epoch) 更新策略，同时训练价值网络
  4. 丢弃旧数据，回到第 1 步（on-policy 特性）

Acrobot 环境说明：
  - 状态空间: s = [cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), θ̇₁, θ̇₂]  (6维连续)
  - 动作空间: a ∈ {0, 1, 2}  (3个离散动作：施加 +1/0/-1 扭矩)
  - 奖励函数: 每走一步奖励 -1，目标是尽快将末端摆过横线
  - 终止条件: 末端超过横线 或 达到最大步数 (500步)

参考论文:
  Schulman et al., "Proximal Policy Optimization Algorithms", 2017
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical  # 用于离散动作空间的类别分布
import gymnasium as gym
import matplotlib.pyplot as plt

# ==================== 超参数 ====================
SEED = 42  # 随机种子，保证实验可复现
ENV_NAME = "Acrobot-v1"  # Gymnasium 环境名称
# 自动选择设备：有 GPU 用 GPU，否则用 CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99  # 折扣因子 γ：未来奖励的衰减系数，越接近1越重视长期回报
LAMBDA = 0.95  # GAE 的 λ 参数：平衡偏差与方差，λ=1 退化为蒙特卡洛，λ=0 退化为 TD(0)
LR = 3e-4  # Adam 优化器学习率
STEPS_PER_UPDATE = 2048  # 每次策略更新前，在环境中采集的总步数（rollout buffer 大小）
TRAIN_EPOCHS = 10  # 每次收集数据后，对同一批数据训练的轮数（PPO 允许多轮复用）
MINIBATCH_SIZE = 64  # 每个梯度更新步的小批量大小
CLIP_EPS = 0.2  # PPO 截断参数 ε：限制策略比率 r(θ) 在 [1-ε, 1+ε] 范围内
VALUE_COEF = 0.5  # 总损失中 Critic (价值函数) 损失的权重系数
ENTROPY_COEF = 0.01  # 总损失中熵奖励的权重系数（鼓励探索，防止策略过早收敛）
TOTAL_UPDATES = (
    150  # 总的策略更新次数（外层循环），总环境步数 = TOTAL_UPDATES × STEPS_PER_UPDATE
)


# ==================== 工具函数 ====================
def set_seed(seed):
    """
    设置所有随机源的种子，确保实验可复现。
    包括：Python 内置 random、NumPy、PyTorch (CPU & GPU)。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae(rewards, dones, values, next_value):
    """
    计算 GAE (Generalized Advantage Estimation) 优势估计。

    GAE 通过指数加权平均不同步数的 TD 误差，在偏差和方差之间取得平衡：

    公式：
      δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)        # TD 误差
      A_t^GAE = δ_t + (γλ) * mask * A_{t+1}^GAE          # 递推计算优势

    其中 mask = 1 - done，当 episode 终止时截断 bootstrap。

    参数:
        rewards    (list[float]): 每一步的即时奖励 r_t，长度为 T
        dones      (list[float]): 每一步是否终止 (0.0 或 1.0)，长度为 T
        values     (list[float]): Critic 网络估计的 V(s_t)，长度为 T
        next_value (float):       最后一个状态 s_T 的价值估计 V(s_T)，用于 bootstrap

    返回:
        advantages (np.ndarray): GAE 优势估计 A_t^GAE，形状 (T,)
        returns    (np.ndarray): 目标回报值 = A_t + V(s_t)，用于训练 Critic，形状 (T,)
    """

    advantages = []  # 存储每一步的优势估计
    gae = 0.0  # 从时间步 T-1 向前递推，初始 GAE 为 0

    # 将 next_value 追加到 values 末尾，方便索引 values[t+1]
    # values 现在长度为 T+1：[V(s_0), V(s_1), ..., V(s_{T-1}), V(s_T)]
    values = values + [next_value]

    # 从后向前遍历（reversed），因为 A_t 依赖于 A_{t+1}
    for t in reversed(range(len(rewards))):
        # mask: episode 终止时为 0，阻断未来回报的传播；未终止时为 1
        mask = 1.0 - dones[t]

        # TD 误差 δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
        # 当 done=True 时，mask=0，V(s_{t+1}) 不参与计算（因为 episode 已结束）
        delta = rewards[t] + GAMMA * values[t + 1] * mask - values[t]

        # GAE 递推：A_t = δ_t + γ * λ * mask * A_{t+1}
        # γλ 控制了对更远未来 TD 误差的衰减速度
        gae = delta + GAMMA * LAMBDA * mask * gae

        # 插入到列表头部（因为是从后往前算的）
        advantages.insert(0, gae)

    advantages = np.array(advantages, dtype=np.float32)

    # returns = A_t + V(s_t)，即 TD(λ) 回报估计
    # 这是 Critic 的训练目标：让 V(s_t) 逼近 returns
    returns = advantages + np.array(values[:-1], dtype=np.float32)

    return advantages, returns


# ==================== Actor-Critic 网络 ====================
class ActorCritic(nn.Module):
    """
    Actor-Critic 共享骨干网络。

    网络结构：
      输入 (obs_dim) → Linear(64) → Tanh → Linear(64) → Tanh → 骨干特征 h
                                                                  ├─→ Actor head:  Linear(act_dim)  → logits (未归一化的动作概率)
                                                                  └─→ Critic head: Linear(1)        → V(s)  (状态价值估计)

    Actor 和 Critic 共享骨干网络 (backbone)：
      - 优点：参数共享减少计算量，底层特征可复用
      - Actor head 输出 logits，通过 Categorical 分布采样动作
      - Critic head 输出标量 V(s)，估计当前状态的期望回报
    """

    def __init__(self, obs_dim, act_dim):
        """
        参数:
            obs_dim (int): 观测空间维度（Acrobot 为 6）
            act_dim (int): 动作空间大小（Acrobot 为 3）
        """
        super().__init__()

        # 共享骨干网络：两层全连接 + Tanh 激活
        # 使用 Tanh 而非 ReLU 是 RL 中的常见选择，因为 Tanh 输出有界 [-1, 1]，
        # 有助于稳定训练
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, out_features=64),  # 输入层 → 64 维隐藏层
            nn.Tanh(),  # 激活函数
            nn.Linear(in_features=64, out_features=64),  # 64 → 64 隐藏层
            nn.Tanh(),  # 激活函数
        )

        # Actor head（策略网络）：输出每个动作的 logit 值
        # logits 经过 softmax 后变成概率分布 π(a|s)
        self.actor = nn.Linear(in_features=64, out_features=act_dim)

        # Critic head（价值网络）：输出标量 V(s)
        self.critic = nn.Linear(in_features=64, out_features=1)

    def forward(self, obs):
        """
        前向传播：输入观测，输出动作 logits 和状态价值。

        参数:
            obs (Tensor): 观测张量，形状 (batch_size, obs_dim)

        返回:
            logits (Tensor): 动作 logits，形状 (batch_size, act_dim)
            value  (Tensor): 状态价值 V(s)，形状 (batch_size,)
        """
        h = self.backbone(obs)  # 提取共享特征，形状 (batch_size, 64)

        logits = self.actor(h)  # Actor 输出 logits，形状 (batch_size, act_dim)
        value = self.critic(h).squeeze(-1)  # Critic 输出标量，squeeze 去掉最后一维
        # (batch_size, 1) → (batch_size,)

        return logits, value

    def get_action(self, obs):
        """
        根据当前策略采样一个动作（用于数据采集阶段）。

        流程：
          1. 将 numpy 观测转为 Tensor 并送入网络
          2. 用 Categorical 分布从 logits 中采样动作
          3. 记录该动作的 log 概率（后续计算策略比率 r(θ) 时需要）

        参数:
            obs (np.ndarray): 当前状态观测，形状 (obs_dim,)

        返回:
            action   (int):   采样的动作索引
            log_prob (float): 该动作在当前策略下的 log 概率 log π_θ(a|s)
            value    (float): 当前状态的价值估计 V(s)
        """
        # 将 numpy 数组转为 Tensor，添加 batch 维度 (1, obs_dim)
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(
            0
        )  # (obs_dim,) → (1, obs_dim)

        # 采集阶段不需要计算梯度，节省内存和计算
        with torch.no_grad():
            logits, value = self.forward(obs)
            # 用 logits 构建类别分布（内部会自动做 softmax）
            dist = Categorical(logits=logits)

            action = dist.sample()  # 按概率采样动作
            log_prob = dist.log_prob(action)  # 计算 log π_θ(a|s)

            # .item() 将单元素 Tensor 转为 Python 标量
            return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, obs, actions):
        """
        评估一批 (obs, action) 对在当前策略下的 log 概率和价值（用于 PPO 更新阶段）。

        与 get_action 的区别：
          - get_action: 采集时用，一次处理一个状态，不需要梯度
          - evaluate_actions: 更新时用，批量处理，需要梯度来反向传播

        参数:
            obs     (Tensor): 批量观测，形状 (batch_size, obs_dim)
            actions (Tensor): 批量动作，形状 (batch_size,)

        返回:
            log_probs (Tensor): 每个动作的 log 概率 log π_θ(a|s)，形状 (batch_size,)
            entropy   (Tensor): 策略分布的熵 H(π)，形状 (batch_size,)
                                熵越大表示策略越"随机"，用于鼓励探索
            values    (Tensor): 价值估计 V(s)，形状 (batch_size,)
        """
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)  # 计算给定动作的 log 概率
        entropy = dist.entropy()  # 计算分布的熵 H(π) = -Σ π(a) log π(a)

        return log_probs, entropy, values


# ==================== PPO 训练主循环 ====================
if __name__ == "__main__":
    # ===== 初始化 =====
    set_seed(SEED)  # 固定随机种子

    env = gym.make(ENV_NAME)  # 创建 Acrobot 环境
    obs, _ = env.reset(seed=SEED)  # 重置环境并获取初始观测

    # 获取环境的观测/动作空间维度
    # 使用 assert 帮助类型检查器推断 space 的具体类型
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Discrete)
    obs_dim = env.observation_space.shape[0]  # 观测维度 = 6
    act_dim = int(env.action_space.n)  # 动作数量 = 3

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)  # 创建网络并放到设备上
    optimizer = optim.Adam(model.parameters(), lr=LR)  # Adam 优化器

    episode_return = 0.0  # 当前 episode 的累积回报（float 类型，与 reward 一致）
    all_returns = []  # 记录所有已完成 episode 的回报，用于绘图

    # ===== 外层循环：每次收集一批数据，然后更新策略 =====
    # PPO 是 on-policy 算法：每次更新使用的数据都是由当前策略采集的
    for update in range(TOTAL_UPDATES):

        # ---------- 阶段 1: 采集经验数据 (Rollout) ----------
        # 用当前策略 π_θ 在环境中交互 STEPS_PER_UPDATE 步，收集轨迹数据
        obs_buf = []  # 存储每步的观测 s_t
        act_buf = []  # 存储每步的动作 a_t
        logp_buf = []  # 存储每步的旧策略 log 概率 log π_θ_old(a_t|s_t)
        reward_buf = []  # 存储每步的即时奖励 r_t
        done_buf = []  # 存储每步的终止标志 (0.0 或 1.0)
        value_buf = []  # 存储每步的价值估计 V(s_t)

        for _ in range(STEPS_PER_UPDATE):
            # 用当前策略采样动作，同时获取 log 概率和价值估计
            action, log_prob, value = model.get_action(obs)

            # 在环境中执行动作，获取下一状态、奖励、终止信号
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = (
                terminated or truncated
            )  # Gymnasium 将终止分为 terminated 和 truncated

            # 将当前步的数据存入 buffer
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            reward_buf.append(reward)
            done_buf.append(float(done))
            value_buf.append(value)

            episode_return += float(reward)  # 累加当前 episode 回报
            obs = next_obs  # 转移到下一状态

            # 如果 episode 结束，记录回报并重置环境
            if done:
                all_returns.append(episode_return)
                episode_return = 0.0
                obs, _ = env.reset()

        # ---------- 阶段 2: 计算 GAE 优势估计 ----------
        # 需要最后一个状态 s_T 的价值估计来做 bootstrap
        with torch.no_grad():
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            _, next_value = model(obs_tensor)  # 获取 V(s_T)
            next_value = next_value.item()  # Tensor → float

        # 用 GAE 公式计算优势和回报
        advantages, returns = compute_gae(reward_buf, done_buf, value_buf, next_value)

        # ---------- 阶段 3: 将 buffer 数据转换为 Tensor ----------
        # 后续 PPO 更新需要在 GPU 上进行张量运算
        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=DEVICE)
        act_tensor = torch.tensor(np.array(act_buf), dtype=torch.long, device=DEVICE)
        old_logp_tensor = torch.tensor(
            np.array(logp_buf), dtype=torch.float32, device=DEVICE
        )
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        # 优势标准化：减均值除标准差
        # 这是一个重要的工程技巧，能显著提升训练稳定性
        # 标准化后优势均值为0、方差为1，避免不同 episode 之间优势尺度差异过大
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        data_size = len(obs_buf)  # 数据总量 = STEPS_PER_UPDATE

        # ---------- 阶段 4: PPO 策略更新 ----------
        # PPO 的关键特性：同一批数据可以训练多个 epoch（与普通 PG 只能用一次不同）
        # 这是因为 PPO-Clip 机制限制了每次更新的幅度，防止策略偏离太远
        model.train()  # 设置为训练模式

        for _ in range(TRAIN_EPOCHS):
            # 每个 epoch 随机打乱数据顺序
            indices = np.arange(data_size)
            np.random.shuffle(indices)

            # 按 minibatch 遍历所有数据
            for start in range(0, data_size, MINIBATCH_SIZE):
                # 取出当前 minibatch 的索引和对应数据
                mb_idx = indices[start : start + MINIBATCH_SIZE]
                mb_obs = obs_tensor[mb_idx]  # 小批量观测
                mb_act = act_tensor[mb_idx]  # 小批量动作
                mb_old_logp = old_logp_tensor[mb_idx]  # 旧策略的 log π_old(a|s)
                mb_adv = adv_tensor[mb_idx]  # 小批量优势
                mb_ret = ret_tensor[mb_idx]  # 小批量目标回报

                # 用当前（新）策略重新评估这批数据
                new_logp, entropy, value = model.evaluate_actions(mb_obs, mb_act)

                # 计算策略比率 r(θ) = π_θ(a|s) / π_θ_old(a|s)
                # 在 log 空间做减法等价于概率空间做除法：exp(log_new - log_old) = new/old
                ratio = torch.exp(new_logp - mb_old_logp)

                # ---- PPO-Clip 目标函数 ----
                # L^CLIP = E[min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)]
                #
                # 未截断项：直接用比率乘以优势
                unclipped = ratio * mb_adv

                # 截断项：将比率限制在 [1-ε, 1+ε] 范围内
                # 当 A > 0（好动作）时，clip 防止 r(θ) 变得太大（策略变化过大）
                # 当 A < 0（差动作）时，clip 防止 r(θ) 变得太小
                clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv

                # Actor 损失：取 min 后取负号（因为我们要最大化目标，但优化器做最小化）
                actor_loss = -torch.min(unclipped, clipped).mean()

                # Critic 损失：均方误差 L_V = 0.5 * (V(s) - R_target)²
                # 让价值网络的预测 V(s) 逼近 GAE 计算的目标回报
                critic_loss = 0.5 * (mb_ret - value).pow(2).mean()

                # 熵损失：-H(π) = -E[-log π]
                # 加上熵正则项鼓励策略保持一定的随机性，防止过早确定性收敛
                # 注意这里是负号，因为我们想 *最大化* 熵（即最小化 -H）
                entropy_loss = -entropy.mean()

                # 总损失 = Actor 损失 + c₁ * Critic 损失 + c₂ * 熵损失
                loss = (
                    actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
                )

                # 梯度更新三步曲
                optimizer.zero_grad()  # 清空旧梯度
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 用 Adam 更新参数

        # ---------- 阶段 5: 日志输出 ----------
        if len(all_returns) > 0:
            recent = all_returns[-10:] if len(all_returns) >= 10 else all_returns
            print(
                f"Update {update + 1}/{TOTAL_UPDATES} | "
                f"Episodes: {len(all_returns)} | "
                f"Mean Return (last 10): {np.mean(recent):.1f} | "
                f"Latest: {all_returns[-1]:.1f}"
            )

    env.close()  # 关闭环境，释放资源

    # ==================== 绘制训练曲线 ====================
    plt.figure(figsize=(10, 5))
    # 原始 episode 回报（半透明蓝色线，显示波动）
    plt.plot(all_returns, alpha=0.3, label="Episode Return")
    # 滑动平均线（平滑趋势，更容易看出学习进展）
    if len(all_returns) >= 10:
        window = 10
        smoothed = np.convolve(all_returns, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, len(all_returns)),
            smoothed,
            label=f"Moving Avg ({window})",
        )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("PPO on Acrobot-v1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "/Users/jet/Desktop/LLM_Learning/cs336/code_to_LLM/ppo_acrobot_training.png",
        dpi=150,
    )
    plt.show()
    print("训练完成！曲线已保存。")
