"""
PPO-RLHF 算法标准实现 — 基于大模型(LLM)偏好对齐主流架构的极简仿真
========================================================================

本文件提供了一个基于 PyTorch 的、完全自包含且可运行的标准 PPO-RLHF 训练流程。
它体现了当前工业界主流大模型强化学习对齐（如 OpenRLHF, DeepSpeed-Chat, Hugging Face TRL 等）的核心逻辑。

核心技术特性（主流 PPO-RLHF 实现的标配）：
  1. 四模型架构分离：Actor（策略）、Reference（参考）、Critic（价值）与 Reward（奖励）独立存在。
  2. 标准 Token 级 KL 惩罚奖励：KL 散度作为负反馈加入每一步 Token 的即时奖励，直接输入给 GAE 优势计算。
  3. GAE 优势与目标回报估计：对包含 KL 散度的混合奖励计算 GAE，用于引导策略更新和价值网络训练。
  4. 价值截断（Value Clipping）：防止 Critic 价值预测由于梯度更新过快导致剧烈摆动。
  5. 序列掩码处理（Sequence Masking）：支持变长序列的 Padding 与位置对齐。

仿真模型配置：
  为了实现代码在没有 GPU 集群和海量显存的环境下完全可运行且快速收敛，我们实现了一个轻量级的小型 Transformer 模型：
  - 词表大小（Vocabulary Size）: 128 (支持基本文本生成仿真)
  - 动作空间: 选择下一个 Token
  - 即时奖励: 只有在回答结束（EOS）或生成最大长度时，通过 Reward 网络输出偏好分数并扣除整个生成序列的 KL 惩罚。
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ==================== 超参数 ====================
SEED = 42
VOCAB_SIZE = 128        # 词表大小
D_MODEL = 64            # 模型隐藏层特征维度
NUM_HEADS = 2           # Transformer 注意力头数
NUM_LAYERS = 2          # Transformer 编码器层数
MAX_SEQ_LEN = 32        # 最大总长度（Prompt + Response）

GAMMA = 1.0             # 折扣因子 gamma
LAMBDA = 0.95           # GAE 的 lambda 参数
KL_COEF = 0.1           # KL 惩罚系数 beta
CLIP_EPS = 0.2          # PPO-Clip 截断系数 epsilon
CLIP_VAL = 0.2          # Critic 价值截断范围
LR_ACTOR = 5e-5         # Actor 学习率
LR_CRITIC = 1e-4        # Critic 学习率
TOTAL_UPDATES = 20      # 总交互轮数（外层循环）
STEPS_PER_UPDATE = 32   # 每轮交互采集的样本批次大小 (Batch Size)
PPO_EPOCHS = 4          # 每批数据重复更新次数
MINIBATCH_SIZE = 8      # 小批量梯度下降大小

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==================== 仿真 Transformer LM 结构 ====================
class ToyTransformerLM(nn.Module):
    """
    轻量级的自回归 GPT 式 Transformer 语言模型，用于仿真大模型。
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_len):
        super().__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
        # 使用 PyTorch 标准 Transformer Decoder 模块
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 语言模型 Head，用于预测下一个 Token 的概率分布
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def _generate_causal_mask(self, sz):
        # 产生自回归掩码，防止当前步看到未来 Token
        mask = torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input_ids):
        b, seq_len = input_ids.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0).repeat(b, 1)
        
        # 词嵌入 + 位置嵌入
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        
        # 产生掩码
        causal_mask = self._generate_causal_mask(seq_len)
        
        # 这里 Transformer Decoder 将序列自身作为 target 和 memory 进行自回归编码
        out = self.transformer(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)
        logits = self.lm_head(out)
        return logits


class CriticModel(nn.Module):
    """
    Critic 价值网络：输入完整的序列（Prompt + Response），预测每个 Response Token 步的期望未来总回报。
    """
    def __init__(self, base_model, d_model):
        super().__init__()
        self.base_model = base_model
        # 价值输出头，输出维度为 1 维标量
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        # 复用基础 Transformer 模型提取特征
        b, seq_len = input_ids.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0).repeat(b, 1)
        x = self.base_model.embedding(input_ids) + self.base_model.pos_embedding(pos)
        causal_mask = self.base_model._generate_causal_mask(seq_len)
        out = self.base_model.transformer(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)
        
        # 输出每个位置的状态价值 V(s_t)
        values = self.value_head(out).squeeze(-1)
        return values


class RewardModel(nn.Module):
    """
    仿真 Reward 网络：输入完整的生成序列，预测一个代表人类偏好得分的整体标量分数。
    """
    def __init__(self, base_model, d_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        b, seq_len = input_ids.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0).repeat(b, 1)
        x = self.base_model.embedding(input_ids) + self.base_model.pos_embedding(pos)
        causal_mask = self.base_model._generate_causal_mask(seq_len)
        out = self.base_model.transformer(x, x, tgt_mask=causal_mask, memory_mask=causal_mask)
        
        # 仿真打分：取出最后一个位置的表征进行线性映射，作为整句得分
        last_step_features = out[:, -1, :]
        scores = self.reward_head(last_step_features).squeeze(-1)
        return scores


# ==================== 仿真数据流：自回归生成 ====================
def generate_response(actor, prompts, gen_len):
    """
    自回归生成 Response Token 并收集 log_probs。
    """
    actor.eval()
    input_ids = prompts.clone()
    b = input_ids.size(0)
    
    # 存储生成的 Response 每一个 Token 的 Log 概率
    log_probs = []
    
    for _ in range(gen_len):
        with torch.no_grad():
            logits = actor(input_ids)
            # 提取最后一步预测下一个 Token 的概率分布
            next_token_logits = logits[:, -1, :]
            dist = Categorical(logits=next_token_logits)
            
            # 采样下一个动作（即 Token）
            next_token = dist.sample()
            log_prob = dist.log_prob(next_token)
            
            # 追加到序列中继续自回归
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            log_probs.append(log_prob)
            
    # 将列表拼接成 shape 为 (BatchSize, GenLen) 的张量
    log_probs = torch.stack(log_probs, dim=1)
    response_ids = input_ids[:, prompts.size(1):]
    return input_ids, response_ids, log_probs


# ==================== GAE & 业界主流 KL 惩罚核心计算 ====================
def compute_advantages_and_returns(rewards, values, next_value, pad_mask):
    """
    计算广义优势估计（GAE）与 Critic 拟合目标（Returns）。
    
    数学推导关系传导：
      δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
      A_t^GAE = δ_t + (γλ) * mask * A_{t+1}^GAE
      R_t = A_t + V(s_t)
      
    参数:
      rewards  (Tensor): 包含即时 KL 惩罚的混合奖励序列， shape 为 (BatchSize, GenLen)
      values   (Tensor): Critic 预测的中间价值序列， shape 为 (BatchSize, GenLen)
      next_value (Tensor): 终止状态的 bootstrapping 估计值， shape 为 (BatchSize,)
      pad_mask (Tensor): 用来做 Padding 的掩码，1 表示有效 token，0 表示 padding
    """
    b, gen_len = rewards.size()
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(b, device=DEVICE)
    
    # 从后向前反向递推优势
    for t in reversed(range(gen_len)):
        # 如果当前位置是有效生成则 mask=1，否则若已处于 padding 区则 mask=0
        mask = pad_mask[:, t]
        
        # 寻找下一步的预测价值，若是最后一步，则使用 next_value 补充
        nv = values[:, t + 1] if t < gen_len - 1 else next_value
        
        # TD 误差 δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
        delta = rewards[:, t] + GAMMA * nv * mask - values[:, t]
        
        # GAE 递推计算优势值： A_t = δ_t + γ * λ * mask * A_{t+1}
        gae = delta + GAMMA * LAMBDA * mask * gae
        advantages[:, t] = gae * mask
        
    # 计算价值网络训练拟合的目标标签 R̂_t = A_t + V_old(s_t)
    returns = advantages + values
    return advantages, returns


# ==================== 训练单步 PPO-RLHF 核心流程 ====================
def train_ppo_step(actor, ref, critic, reward_model, prompts, gen_len, actor_opt, critic_opt):
    actor.eval()
    ref.eval()
    critic.eval()
    reward_model.eval()
    
    # ---------- 1. Rollout 阶段：自回归采样 ----------
    b = prompts.size(0)
    prompt_len = prompts.size(1)
    
    # 生成完整序列（Prompt + Response），动作 token IDs 以及 Actor 旧对数概率
    full_seq, response_ids, old_log_probs = generate_response(actor, prompts, gen_len)
    
    # 仿真生成掩码，假设 0 是 padding 填充（此案例为固定长度无 padding 演示，mask 设为全 1）
    pad_mask = torch.ones(b, gen_len, device=DEVICE)
    
    # ---------- 2. 业界主流 KL 散度与奖励计算（重点！！！） ----------
    # 核心：KL 散度必须作为 Token 级别的负惩罚，直接加入到优势计算的奖励机制中
    with torch.no_grad():
        # 获取 SFT 参考模型在对应 Response Token 上的对数概率
        # logits_ref 的 shape 为 (BatchSize, PromptLen+GenLen-1, VocabSize)
        logits_ref = ref(full_seq)[:, :-1, :]
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)
        # 获取对应 Response 动作的 log π_ref(a|s)
        log_probs_ref_response = log_probs_ref[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
        old_log_probs_ref = log_probs_ref_response.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
        
        # 计算外部打分系统的奖励分数 (Reward Model 打分)
        # Mainstream Reward Model 对完整的整个 Response (加上上下文 Prompt) 打出序列级别分数
        reward_scores = reward_model(full_seq)  # shape 为 (BatchSize,)
        
        # 计算即时奖励序列 (Token-level Rewards)
        # 标准主流公式：
        # - 中间步 Token 即时奖励只含 KL 惩罚： r_t = -β * (log π_new - log π_ref)
        # - 结束 Token (last_step) 即时奖励：  r_T = Reward_Score - β * (log π_new - log π_ref)
        token_kl_penalty = -KL_COEF * (old_log_probs - old_log_probs_ref)  # shape 为 (BatchSize, GenLen)
        
        token_rewards = token_kl_penalty.clone()
        # 将外部偏好奖励评分 Reward Model Score 注入到整个生成序列的最后一个有效 Token 的奖励中
        token_rewards[:, -1] += reward_scores
        
        # 获取 Critic 价值网络对当前 rollout 序列的中间状态估值
        # values_seq 的 shape 为 (BatchSize, PromptLen+GenLen-1)
        values_seq = critic(full_seq)
        # 对应位置抽取针对 Response 每一个 token 动作发出时的状态价值 V_old(s_t)
        old_values = values_seq[:, prompt_len-1:prompt_len-1+gen_len]  # shape 为 (BatchSize, GenLen)
        next_value = torch.zeros(b, device=DEVICE)  # 终止位置 bootstrap 值定为 0
        
    # ---------- 3. 计算 GAE 优势与目标回报 (Advantages & Returns) ----------
    advantages, returns = compute_advantages_and_returns(token_rewards, old_values, next_value, pad_mask)
    
    # 优势归一化（业界主流工程实践，可以极大地稳定策略更新）
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages_normalized = (advantages - adv_mean) / (adv_std + 1e-8)
    
    # ---------- 4. PPO 多轮优化更新阶段 (Optimization) ----------
    actor.train()
    critic.train()
    
    epoch_actor_loss = 0.0
    epoch_critic_loss = 0.0
    epoch_approx_kl = 0.0
    
    # 对同一批 Rollout 交互数据，循环训练多轮 (PPO Epochs)
    for ppo_epoch in range(PPO_EPOCHS):
        # 打乱 Batch 的样本索引顺序
        indices = torch.randperm(b)
        
        for start_idx in range(0, b, MINIBATCH_SIZE):
            end_idx = start_idx + MINIBATCH_SIZE
            mb_idx = indices[start_idx:end_idx]
            
            # 抽取当前的小批量数据
            mb_full_seq = full_seq[mb_idx]
            mb_response_ids = response_ids[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_old_values = old_values[mb_idx]
            mb_advantages = advantages_normalized[mb_idx]
            mb_returns = returns[mb_idx]
            
            # --- 4.1 Actor 更新过程 ---
            logits_new = actor(mb_full_seq)[:, :-1, :]
            log_probs_new = F.log_softmax(logits_new, dim=-1)
            # 计算新策略下的对数概率 log π_θ(a|s)
            log_probs_new_response = log_probs_new[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
            new_log_probs = log_probs_new_response.gather(2, mb_response_ids.unsqueeze(-1)).squeeze(-1)
            
            # 计算比率 ratio = exp(new_logp - old_logp) = π_new / π_old
            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            
            # PPO-Clip 目标损失计算
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
            
            # 注意：主流 PPO-RLHF 因为 KL 惩罚已经进入到优势优势计算 A(t) 中，
            # 策略损失函数本身只含单纯的 PPO-Clip 代理损失（不含有独立的 Actor KL 损失项）！
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # --- 4.2 Critic 更新过程（含主流 Value Clipping） ---
            new_values_full = critic(mb_full_seq)
            new_values = new_values_full[:, prompt_len-1:prompt_len-1+gen_len]
            
            # 未截断的 Critic 损失：平方差均值
            val_loss_unclipped = (new_values - mb_returns).pow(2)
            
            # 价值变化截断：限制预测价值的变化不要比 old_values 漂移得太远
            mb_values_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -CLIP_VAL, CLIP_VAL)
            val_loss_clipped = (mb_values_clipped - mb_returns).pow(2)
            
            # Critic 损失取两者中的最大值，从而约束参数不会过大跳转
            critic_loss = 0.5 * torch.max(val_loss_unclipped, val_loss_clipped).mean()
            
            # --- 4.3 梯度优化与更新 ---
            # 训练 Actor
            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()
            
            # 训练 Critic
            critic_opt.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
            critic_opt.step()
            
            # 收集运行统计信息
            with torch.no_grad():
                approx_kl = (0.5 * (new_log_probs - mb_old_log_probs).pow(2)).mean()
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                epoch_approx_kl += approx_kl.item()
                
    # 计算均值
    num_updates = PPO_EPOCHS * math.ceil(b / MINIBATCH_SIZE)
    return (
        epoch_actor_loss / num_updates,
        epoch_critic_loss / num_updates,
        epoch_approx_kl / num_updates,
        reward_scores.mean().item(),
        token_kl_penalty.mean().item()
    )


# ==================== 主控运行测试函数 ====================
def main():
    setup_seed(SEED)
    print("====== 主流大模型 PPO-RLHF 仿真对齐引擎启动 ======")
    print(f"当前训练设备: {DEVICE}")
    
    # 1. 实例化核心组件（大模型训练的四模型架构）
    # 策略模型 (Policy/Actor)
    actor_base = ToyTransformerLM(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(DEVICE)
    actor = actor_base
    
    # 冻结的 SFT 参考模型 (Reference Model)
    ref = ToyTransformerLM(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(DEVICE)
    ref.load_state_dict(actor.state_dict())  # 初始状态下 Policy 和 Reference 参数完全一致
    ref.requires_grad_(False)
    
    # 价值估计模型 (Critic/Value Model)
    critic_base = ToyTransformerLM(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(DEVICE)
    critic = CriticModel(critic_base, D_MODEL).to(DEVICE)
    
    # 人类偏好打分模型 (Reward Model)
    reward_base = ToyTransformerLM(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(DEVICE)
    reward_model = RewardModel(reward_base, D_MODEL).to(DEVICE)
    reward_model.requires_grad_(False)  # 奖励模型在 RLHF 过程中保持冻结
    
    # 2. 初始化优化器（Actor 和 Critic 各自独立维护各自的学习率和更新参数）
    actor_opt = optim.AdamW(actor.parameters(), lr=LR_ACTOR)
    critic_opt = optim.AdamW(critic.parameters(), lr=LR_CRITIC)
    
    # 3. 构造仿真 Prompt 数据源
    # 假设输入 Prompt 长度为 8，每次随机抽取一批 prompts
    prompt_len = 8
    gen_len = 10  # 自动生成 10 个 token 的 Response
    
    print("\n开始 PPO-RLHF 对齐微调流程...")
    for step in range(TOTAL_UPDATES):
        # 随机采样生成 Prompt 数据 (BatchSize=STEPS_PER_UPDATE, Length=prompt_len)
        prompts = torch.randint(1, VOCAB_SIZE, (STEPS_PER_UPDATE, prompt_len), device=DEVICE)
        
        # 执行一轮标准的 PPO 收集并训练更新
        a_loss, c_loss, approx_kl, mean_reward, mean_kl_penalty = train_ppo_step(
            actor, ref, critic, reward_model, prompts, gen_len, actor_opt, critic_opt
        )
        
        # 打印训练统计指标
        # mean_kl_penalty 对应： -β * (log π_new - log π_ref) 的平均水平
        # 偏离度越低越好，当对齐发生作用后，Mean Reward 分数应该呈上升态势
        print(f"轮数 [{step+1}/{TOTAL_UPDATES}] | "
              f"Actor Loss: {a_loss:.4f} | "
              f"Critic Loss: {c_loss:.4f} | "
              f"Approx KL: {approx_kl:.5f} | "
              f"RM Score: {mean_reward:.4f} | "
              f"KL Penalty: {mean_kl_penalty:.4f}")
        
    print("\nPPO-RLHF 对齐训练结束！策略与价值网络已更新收敛。")

if __name__ == "__main__":
    main()
