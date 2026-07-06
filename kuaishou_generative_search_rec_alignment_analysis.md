# 快手生成式搜推系列论文 CPT、SFT、RL 样本构造与对齐策略深度分析

在生成式搜推（Generative Search & Recommendation）中，离散化的语义 ID (SID) 仅仅解决了“如何表示物料”的问题。而“如何训练模型”来理解这些 SID、根据用户行为进行精准检索与排序，则依赖于持续预训练（CPT）、监督微调（SFT）和强化学习（RL）阶段的样本构造与对齐格式。

本文对快手搜推技术栈在这三个阶段的样本构造、格式设计及对齐演进进行深度拆解。

---

## 一、搜推对齐技术全景图

在快手的生成式搜推演进中，三大训练阶段（CPT、SFT、RL）的定位和样本构造方式经历了显著的范式转移：

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 持续预训练 (CPT) 阶段                                                   │
│ 从“单向语义对齐 (OneSearch)” ──► “多模态/多层次语义接地 (OpenOneRec/OneReason)” │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 监督微调 (SFT) 阶段                                                    │
│ 从“Query-Item Co-occurrence预测” ──► “三级认知增强的CoT推理机制 (OneReason)”│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 强化学习与偏好对齐 (RL/Alignment) 阶段                                  │
│ 从“点/列表级超轻量级DPO” ──► “GBPO自适应剪切” ──► “TPMA-GRPO细粒度信用分配”│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 二、各论文 CPT/SFT/RL 样本构造与格式详解

### 2.1 OneSearch (2509.03236) — 课程化多阶段 SFT 与自适应 PARS 机制

OneSearch 是首个在工业级搜索场景部署 of 生成式框架。其核心思想是打破传统级联架构，用自适应奖励机制（PARS）约束生成过程。

#### 1. 持续预训练（CPT）与多阶段监督微调（SFT）样本构造
OneSearch 将 SFT 拆分为由易到难的三个课程化（Curriculum）阶段：
*   **阶段一：语义内容对齐 (Semantic Alignment)**
    *   **样本构造**：物料标题 $T_i$ 、类目属性与 SID 的映射对。
    *   **格式**：
        *   **Text-to-SID 格式**：
        ```text
        Input: "Predict the semantic ID of the item: [Item Title]"
        Output: "<sid_level_1_x><sid_level_2_y><sid_level_3_z>"
        ```
        *   **SID-to-Text 格式**：
        ```text
        Input: "Translate the semantic ID to text: <sid_level_1_x><sid_level_2_y><sid_level_3_z>"
        Output: "[Item Title]"
        ```
        *   **属性/类目预测格式**：
        ```text
        Input: "Predict the category and attributes of the item: [Item Title]"
        Output: "Category: [Category] | Brand: [Brand] | Attributes: [Attr1, Attr2]"
        ```
    *   **目的**：让大模型理解离散 Token 序列（SID）对应的真实语言含义。
*   **阶段二：共现同步 (Co-occurrence Synchronization)**
    *   **样本构造**：Query-Item 点击对 $(q, i)$ 及其 SID 对。
    *   **格式**：
        ```text
        Input: "Search query: [User Query] -> Predict target item semantic ID:"
        Output: "<sid_level_1_x><sid_level_2_y><sid_level_3_z>"
        ```
        ```text
        Input: "Retrieve search query for item: <sid_level_1_x><sid_level_2_y><sid_level_3_z>"
        Output: "[User Query]"
        ```
    *   **目的**：学习 Query 与商品之间的语义相关性和协同过滤信号。
*   **阶段三：用户个性化建模 (User Personalization Modeling)**
    *   **样本构造**：结合用户画像、长期购买偏好、短期点击流与目标商品 SID。
    *   **滑动窗口数据增强**：对用户的短期行为序列进行滑动窗口切片，用于训练模型感知兴趣的动态转移。
    *   **格式**：
        ```text
        Input: "User Info - Age: [Age], Gender: [Gender], City: [City] | Long-term Pref: [Brand_A, Category_B] | Short-term Behaviors: [Query_1 -> Click_Item_1_SID, Query_2 -> Click_Item_2_SID] | Current Query: [Current Query]"
        Output: "Target Item SID: <sid_level_1_x><sid_level_2_y><sid_level_3_z>"
        ```

#### 2. RL/对齐样本构造与损失函数 (PARS)
*   **样本构造**：引入 **Preference-Aware Reward System (PARS)**，构造 Query 维度下的商品列表偏好样本。
*   **六级行为层级与自适应权重**：
    将用户的交互反馈划分为 6 个等级（例如：购买、收藏、加购、点击、曝光未点击、未曝光），赋予递减的基础权重 $\lambda$ ：

    $$
    \lambda = [\lambda_{\text{buy}}, \lambda_{\text{add-to-cart}}, \lambda_{\text{collect}}, \lambda_{\text{click}}, \lambda_{\text{expose-unclick}}, \lambda_{\text{unexpose}}]
    $$

    具体配置为： $\lambda = [2.0, 1.5, 1.0, 0.5, 0.2, 0.0]$ 。
*   **校准奖励模型 (Reward Model)**：
    为了避免马太效应，融合了近 7 日的 CTR、CVR 和 CTCVR 指标，训练三塔结构预测模型输出偏好分数 $r(q, i)$ ：

    $$
    r(q, i) = w_{\text{ctr}} \cdot P(\text{click} \mid q, i) + w_{\text{cvr}} \cdot P(\text{buy} \mid \text{click}, q, i)
    $$

    用于加权 SFT 的 Loss，或作为 DPO 的软性偏好标签。
*   **Listwise 排序优化 Loss 函数**：

    $$
    \mathcal{L}_{\text{PARS}} = -\sum_{q} \log \frac{\sum_{i \in \mathcal{I}^+} \exp(r(q, i) / \tau)}{\sum_{j \in \mathcal{I}^+ \cup \mathcal{I}^-} \exp(r(q, j) / \tau)}
    $$

    $\mathcal{I}^+$ 代表正反馈商品集合， $\mathcal{I}^-$ 代表曝光未点击等负反馈商品集合， $\tau$ 为温度超参数。

---

### 2.2 OneSearch-V2 (2603.24422) — 潜空间推理与 TPMA-GRPO 细粒度对齐

OneSearch-V2 专注于解决长尾 Query 理解困难和强化学习“奖励作弊（Reward Hacking）”的问题。

#### 1. SFT 阶段：Thought-Augmented 显式推理链构造
*   **样本构造**：通过离线的高能 LLM（如 GPT-4 级）针对“复杂/模糊 Query-用户行为”生成紧凑、提炼的关键词 CoT（Chain of Thought）推理路径，指出用户的真实意图、品类倾向和品牌偏好。
*   **数据格式**：
    ```text
    Prompt: User Info: [Info] | Query: [Complex Query]
    Response: <thought>
    - Query intent parsing: The user is looking for [intent].
    - Preference constraints: The user prefers [brand/category/price-range].
    - Deduction: Recommend items matching both the intent and historical preference.
    </thought>
    Target Item SID: <sid_level_1_x><sid_level_2_y><sid_level_3_z>
    ```
*   **潜空间推理自回归蒸馏 (Latent Reasoning Self-Distillation) 损失函数**：
    为了使学生模型在在线服务时不输出中间的 `<thought>` 词元也能具备推理能力，OneSearch-V2 将带有推理链的教师概率分布蒸馏给不带推理链的当前策略：

    $$
    \mathcal{L}_{\text{distill}} = \text{D}_{\text{KL}} \left( P_{\text{teacher}}(y \mid x, \text{thought}) \parallel \pi_{\theta}(y \mid x) \right)
    $$

    其中 $y$ 为目标商品 SID 序列， $x$ 为用户 Query 及行为序列输入， $\text{thought}$ 为教师模型生成的 CoT 文本。

#### 2. RL 阶段：TPMA-GRPO 信用分配与自适应剪切
*   **在策采样轨迹 (On-Policy Rollout) 与样本构造方法**：
    在 RL 阶段，输入端直接复用 SFT 阶段的 Prompt $x_u$ （即用户画像及搜索 Query）。模型不再匹配静态的点击日志商品，而是使用当前策略模型 $\pi_{\theta}$ 在线对每个 Prompt **在策生成（Rollout）一组共 $G$ 条候选轨迹** $\{y_1, y_2, \dots, y_G\}$ 。每条轨迹包含自生成的 `<thought>` 推理链以及目标商品 SID 序列。这些自生成的轨迹将通过行为反馈函数进行打分来指导策略更新，从而从模型自身的输出中提供纠错信号。
*   **TPMA-GRPO（令牌-位置边际优势）优化目标函数**：

    $$
    \mathcal{L}_{\text{TPMA}} = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{L} \sum_{t=1}^{L} \text{gate}_{i,t} \cdot \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right)
    $$

    其中 $G$ 为 Rollout 样本数量， $L$ 为序列长度， $r_{i,t}(\theta)$ 为当前策略与基准策略的概率比值：

    $$
    r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} \mid x_u, o_{i,\lt t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid x_u, o_{i,\lt t})}
    $$

*   **前缀正确性门控机制 (Prefix Gate)**：

    $$
    \text{gate}_{i,t} = \prod_{j=1}^{t-1} \mathbb{I}(o_{i,j} == o_{i,j}^*)
    $$

    其中 $\mathbb{I}$ 是指示函数， $o_{i,j}^*$ 代表真实的 Ground-Truth Token。如果前 $t-1$ 个 Token 中存在错误，则第 $t$ 步的梯度被完全屏蔽（ $\text{gate}_{i,t} = 0$ ）。
*   **Token 位置级边际优势 $\hat{A}_{i,t}$**：

    $$
    \hat{A}_{i,t} = R_{i, \ge t} - \bar{R}_{\ge t}
    $$

    其中 $R_{i, \ge t}$ 是第 $i$ 个轨迹从位置 $t$ 开始的累积奖励， $\bar{R}_{\ge t}$ 是当前组内所有轨迹在位置 $t$ 开始的平均奖励。

---

### 2.3 OneRec (2506.13695) — 协同对齐预训练与迭代偏好优化 (IPA)

OneRec 将端到端生成式架构引入推荐场景，使用 Encoder-Decoder 结构处理 session-wise（会话级）推荐。

#### 1. 持续预训练（CPT）与 SFT 样本构造
*   **任务构造**：
    *   **I2I 协同损失**：用 Swing 算法计算商品相似度，构造正样本对，在预训练中拉近相似商品的表征。
        Swing 算法相似度计算公式为：

        $$
        w_{i,j} = \sum_{u \in U_i \cap U_j} \sum_{v \in U_i \cap U_j} \frac{1}{\alpha + |I_u \cap I_v|}
        $$

        其中 $U_i$ 和 $U_j$ 分别是交互过商品 $i$ 和 $j$ 的用户集合， $I_u$ 和 $I_v$ 分别是用户 $u$ 和 $v$ 交互过的商品集合， $\alpha$ 是平滑常数。
        样本对输入输出格式为：
        ```text
        Input: "Anchor Item: <sid_level_1_a><sid_level_2_b><sid_level_3_c> | Match similar item:"
        Output: "<sid_level_1_x><sid_level_2_y><sid_level_3_z>"
        ```
    *   **Caption-Gen 损失**：将短视频的多模态特征（图像帧、ASR音频、描述）输入大模型，生成对应的 Item Caption。
        ```text
        Input: "Generate a dense description for the video frame sequence [Frames] and audio transcript [ASR]:"
        Output: "[Detailed Video Caption]"
        ```
*   **对齐格式（SFT）**：
    *   **会话级 NTP 格式**：
        ```text
        Prompt: User history sequence: [SID_1, SID_2, ..., SID_n] | Context: [Time_of_Day, Device]
        Response: [SID_n+1, SID_n+2, ..., SID_n+k]
        ```
        Loss 仅在 Target 部分计算，对 Prompt 部分的 Token 进行 Mask：

        $$
        \mathcal{L}_{\text{NTP}} = -\sum_{t=n+1}^{n+k} \log P(SID_t \mid SID_{\lt t}, \text{Prompt})
        $$

#### 2. RL/DPO 阶段：迭代偏好优化 (IPA)
*   **在策候选采样与偏好样本对 $(q_w, q_l)$ 构造方法**：
    由于推荐系统在线只能观测到单次曝光结果，无法直接构造 DPO 所需的 $(q_w, q_l)$ 对。OneRec 采用自生成的“Self-Hard Negatives”采样构造法：
    1. 输入历史 Prompt $x_u$ ，使用当前策略模型 $\pi_{\theta}$ 通过 **Beam Search** 算法采样生成 $M$ 个候选推荐列表序列 $\{q_1, q_2, ..., q_M\}$ 。
    2. 使用预先训练的多目标用户行为 Reward Model $R(x_u, q_j)$ 对这些自生成的序列进行打分评估。
    3. 提取得分最高和最低的序列对，分别作为胜出样本（Chosen $q_w$ ）和落败样本（Rejected $q_l$ ）：

        $$
        q_w = \arg\max_{q_j} R(x_u, q_j), \quad q_l = \arg\min_{q_j} R(x_u, q_j)
        $$

*   **DPO Loss 函数**：

    $$
    \mathcal{L}_{\text{DPO}}(\theta; \theta_{\text{ref}}) = -\mathbb{E}_{(x_u, q_w, q_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(q_w \mid x_u)}{\pi_{\text{ref}}(q_w \mid x_u)} - \beta \log \frac{\pi_{\theta}(q_l \mid x_u)}{\pi_{\text{ref}}(q_l \mid x_u)} \right) \right]
    $$

---

### 2.4 OneRec-V2 (2508.20900) — 剪切限制与时间感知奖励塑形

OneRec-V2 从 Encoder-Decoder 升级为 **Lazy Decoder-Only**，重点解决了大规模长视频推荐时的时长偏好偏置。

#### 1. SFT 阶段的 Loss 均值化改革
*   **格式调整**：保持 Prompt 拼接格式，但在生成 Loss 计算上，从原本的 3 个 SID Token 的 $\text{Sum}(\text{Loss})$ 调整为 $\text{Mean}(\text{Loss})$ ，防止生成阶段梯度被前几层 SID Token 绑架，使得细粒度商品属性得到同等程度的训练。

    $$
    \mathcal{L}_{\text{SFT-Mean}} = -\frac{1}{3} \sum_{k=1}^{3} \log P(s_k \mid s_{\lt k}, x_u)
    $$

#### 2. RL 阶段的自适应剪切与时长奖励塑形 (GBPO)
*   **时间感知奖励塑形 (Duration-Aware Reward Shaping)**：
    为了消除长视频自然播放时间长、短视频容易被跳过的偏置，将物料按时长进行对数分桶：

    $$
    F(d) = \lfloor \log_{\beta}(d + \epsilon) \rfloor
    $$

    在每个分桶内，对用户的观看时长（Play Time, $pt$ ）进行标准化作为奖励分 $R_{\text{duration}}(pt, d)$ ：

    $$
    R_{\text{duration}}(pt, d) = \frac{pt - \mu_{F(d)}}{\sigma_{F(d)}}
    $$

    其中 $\mu_{F(d)}$ 和 $\sigma_{F(d)}$ 是时长分桶 $F(d)$ 内部所有样本观看时间的均值与标准差。
*   **梯度有界策略优化 (GBPO - Gradient-Bounded Policy Optimization) 剪切比率公式**：

    $$
    \mathcal{L}_{\text{GBPO}}(\theta) = -\mathbb{E} \left[ \min \left( \hat{r} \hat{A}, \text{clip}(\hat{r}, 1 - B(\theta_{\text{old}}), 1 + B(\theta_{\text{old}})) \hat{A} \right) \right]
    $$

    其中 $\hat{r} = \frac{\pi_{\theta}(q_w \mid x_u) / \pi_{\theta}(q_l \mid x_u)}{\pi_{\theta_{\text{old}}}(q_w \mid x_u) / \pi_{\theta_{\text{old}}}(q_l \mid x_u)}$ ，梯度边界定义为：

    $$
    B(\theta_{\text{old}}) = \alpha \cdot \left\| \nabla_{\theta_{\text{old}}} \log \frac{\pi_{\theta_{\text{old}}}(q_w \mid x_u)}{\pi_{\theta_{\text{old}}}(q_l \mid x_u)} \right\|_2
    $$

---

### 2.5 OneRec-Think (2510.11639) — 推荐推理激活与 Think-Ahead 架构

OneRec-Think 首次在推荐场景下探索了“先思考、后推荐”的推理样本构造。

#### 1. SFT 阶段：Reasoning Scaffolding（推理脚手架）
*   **样本构造**：结合用户历史序列，生成包含“反思-分析-归纳-演绎-推荐”的完整推荐 CoT 数据。
*   **数据格式**：
    ```text
    Prompt: The user's historical interaction sequence is [SID_1, SID_2]. Help predict the next item.
    Response: <thought>
    - Reflection: The user historically watched [SID_1_Title] (category: Gourmet) and [SID_2_Title] (category: Travel Vlog).
    - Induction: This indicates a strong preference for lifestyle and food-discovery content.
    - Abduction: Since the current time is 6:00 PM, the user is likely looking for dinner inspiration or relaxing short-form videos.
    - Deduction: A high-quality food reviewing video with a high historical watch time ratio fits this latent state.
    - Recommendation: The model should output the semantic ID representing the gourmet exploration video.
    </thought>
    Recommended Item: <sid_level_1_x><sid_level_2_y><sid_level_3_z>
    ```

#### 2. RL 阶段：多有效性奖励函数 (Multi-Validity Reward Function)
考虑用户兴趣多有效性（即存在多个合理的推荐结果），定义序列级奖励分数：

$$
R_{\text{multi}}(Y) = \frac{1}{|Y|} \sum_{y \in Y} \max_{j \in \mathcal{I}^+} \text{Similarity}(y, j)
$$

    $\text{Similarity}(y, j)$ 为预测推荐 ID $y$ 与用户真实正反馈 ID $j$ 在 Latent 语义空间中的余弦相似度。
*   **部署优化：Think-Ahead 架构**：
    为了避免在线推荐中生成数百个 CoT Token 带来严重的 P99 延迟，Think-Ahead 采用了解耦策略，将 pre-computed 的 `Thought` 作为 Prompt 输入，只让模型自回归生成最终的商品 SID（仅耗费几个 token 的生成时间）。

---

### 2.6 OpenOneRec (2512.24762) — 多任务混合指令与 On-policy 蒸馏对齐

OpenOneRec 是快手开源的搜推大模型，其核心在于如何统一多任务并保留通用 LLM 的世界知识。

#### 1. Co-Pretraining（共预训练）样本构造
*   **样本混合比**：推荐领域行为序列 + 文本密集描述（Dense Captions） + 通用文本数据（按 $7:2:1$ 的比例混合）。
*   **对齐格式**：
    ```text
    Input: "Describe the item with semantic ID <sid_level_1_x><sid_level_2_y><sid_level_3_z> in detail:"
    Output: "Item Title: [Title]. Category: [Category]. Modality Description: A video showing [Content], with tags [Tags]."
    ```

#### 2. SFT 阶段：RecIF-Bench 八大任务指令格式
*   **Task 1: Item Understanding (物料理解)**：
    ```text
    Prompt: Instruction: Analyze the attributes and core features of this item. Item SID: <sid_level_1_x><sid_level_2_y><sid_level_3_z>
    Response: Brand: [Brand] | Category: [Category] | Main color: [Color]
    ```
*   **Task 2: Short Video Recommendation (短视频推荐)**：
    ```text
    Prompt: Instruction: Predict the next video the user is likely to watch based on history. History: [Video_SID_1, Video_SID_2]
    Response: Target Video: <sid_level_1_x><sid_level_2_y><sid_level_3_z>
    ```
*   **Task 3: Ad Recommendation (广告推荐)**：
    ```text
    Prompt: Instruction: Recommend a sponsored item matching the user's current context. Context: [App_ID, Slot_Position] | History: [Item_SID_1]
    Response: Target Ad: <sid_level_1_x><sid_level_2_y><sid_level_3_z>
    ```
*   **Task 4: Product Recommendation (电商商品推荐)**：
    ```text
    Prompt: Instruction: Recommend a product to purchase. User clicks: [Product_SID_1]
    Response: Target Product: <sid_level_1_x><sid_level_2_y><sid_level_3_z>
    ```
*   **Task 5: Label Prediction (标签预测)**：
    ```text
    Prompt: Instruction: Predict the category label the user will click next. Behavior sequence: [SID_1, SID_2]
    Response: Predicted Category: [Category_Name]
    ```
*   **Task 6: Interactive Recommendation (交互式推荐)**：
    ```text
    Prompt: System: You are a recommendation assistant. User: "I want some cooking videos but not dessert." Context: [SID_1(dessert), SID_2(cooking)]
    Response: "Here is a recipe video for you: <sid_level_1_x><sid_level_2_y><sid_level_3_z>"
    ```
*   **Task 7: Label-Conditional Recommendation (标签约束推荐)**：
    ```text
    Prompt: Instruction: Recommend an item. Constraint: Category must be 'Sports'. History: [SID_1, SID_2]
    Response: Target Item: <sid_level_1_x(Sports)><sid_level_2_y><sid_level_3_z>
    ```
*   **Task 8: Recommendation Explanation (推荐理由生成)**：
    ```text
    Prompt: Instruction: Explain why this item is recommended to the user. User History: [SID_1(cooking)] | Recommended Item: <sid_level_1_x><sid_level_2_y><sid_level_3_z>(cooking tool)
    Response: Explanation: "We recommended this cooking tool because you recently watched food preparation videos."
    ```

#### 3. Post-Training RL 对齐与在策蒸馏阶段 (On-Policy Distillation, OPD)
*   **在策轨迹采样 (On-Policy Sampling) 与输入样本构造方法**：
    在 OPD 阶段，训练样本**仅包含 Prompt 集合 $\mathcal{D}_{\text{prompts}}$ ，没有静态的目标 Target 文本**。
    该 Prompt 集合通过对推荐历史序列（70% 推荐行为 Log）与通用语言能力语料（20% 密集多模态描述，10% 自然语言推理任务 Prompt）进行混合构造。
    在训练时，当前学生策略模型 $\pi_{\theta}$ 接收 Prompt $x \in \mathcal{D}_{\text{prompts}}$ ，并在线自回归采样生成推荐或对话路径轨迹 $y \sim \pi_{\theta}$ 。学生生成的这串轨迹 $y$ 被输入教师模型 $\pi_{\phi}$ （Qwen-3-8B-Instruct），以教师模型在这些路径上输出的软概率分布作为优化目标（而不是静态的离散 Token）。
*   **OPD 损失函数 (基于 Reverse KL 散度)**：
    学生模型通过最小化其自身生成的轨迹在教师模型上的 Reverse KL 散度进行优化：

    $$
    \mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}} \left[ \text{D}_{\text{KL}} \left( \pi_{\theta}(y \mid x) \parallel \pi_{\phi}(y \mid x) \right) \right]
    $$

    其中， $\pi_{\phi}$ 是教师模型 $\text{Qwen-3-8B-Instruct}$ 在通用语言/推荐任务下的策略输出概率。
    在多任务联合对齐中，该损失与推荐任务损失合并进行优化：

    $$
    \mathcal{L}_{\text{蒸馏}}(\theta) = \mathcal{L}_{\text{RL-Task}}(\theta) + \gamma \cdot \mathcal{L}_{\text{OPD}}(\theta)
    $$

---

### 2.7 OneReason (2606.06260) — 四阶段感知预训练与三级认知增强 CoT

OneReason 针对“推荐中直接使用 itemic CoT 导致性能倒退”的痛点进行了彻底重构，是当前快手搜推对齐技术的最高峰。

#### 1. CPT 阶段：四粒度渐进式感知预训练 (Perception Alignment)
为了让 SID Token 在大模型内部有深刻的语义理解，在预训练中构造了四个层级的感知对齐样本：
*   **Token Level (Token 粒度) 格式**：
    ```text
    Prompt: Decompose the compound token <sid_level_1_x><sid_level_2_y> into parent category representation:
    Response: Parent Node Category: [Outdoors]
    ```
*   **Item Level (物料粒度) 格式**：
    ```text
    Prompt: Answer based on item token <sid_level_1_x><sid_level_2_y><sid_level_3_z>: What is the brand and utility of this item?
    Response: The brand is Brand_A, and it is a professional tent used for summer camping.
    ```
*   **Relational Level (关系粒度) 格式**：
    ```text
    Prompt: Explain the relationship between Item <Item_A_SID> and Item <Item_B_SID>:
    Response: Users who purchased the Item_A_Name (sleeping bag) also frequently co-purchased Item_B_Name (camping tent) for outdoor activities.
    ```
*   **User Level (用户意图粒度) 格式**：
    ```text
    Prompt: Analyze and summarize the core interests from this chronological item sequence: [<Item_1_SID>, <Item_2_SID>]
    Response: Latent User Interest: [Outdoor Sports / Lightweight Camping]
    ```

#### 2. SFT 阶段：三级认知增强 CoT 样本设计
OneReason 抛弃了无逻辑的 Item 拼接，提出了结构极其严格的 **“三层认知增强 CoT”** 格式：
*   **三级认知增强 SFT 样本格式**：
    ```text
    Prompt: User behavior history: [<Item_1_SID>(Category: Running Shoes), <Item_2_SID>(Category: Sports Watch)]
    Response:
    <thought>
    - Item Relation Induction:
      The user interacted with Item_1 (Running Shoes) and Item_2 (Sports Watch). These items exhibit strong co-occurrence in the "Professional Running/Marathon Preparation" sub-domain.
    - Interest Abstraction:
      The user's latent interest is transitioning from general fitness to structured running training, requiring professional monitoring and gear support.
    - Decision Synthesis:
      Based on the outdoor camping interest and brand loyalty to Brand_X, the next ideal target is an outdoor sleeping bag.
    </thought>
    Target: <Item_target_SID>
    ```

#### 3. RL 阶段：Specialize-then-Unify 训练策略
*   **第一步：垂类专精对齐 (Specialize Phase)**
    在电商、短视频、直播、本地生活广告 4 个场景下，利用特定奖励模型分别训练专精策略：

    $$
    \theta_{\text{domain}} = \arg\max_{\theta} \mathbb{E} [ R_{\text{domain}}(x_u, y) ] - \beta \text{D}_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{sft}})
    $$

*   **第二步：混合专家网络统一融合与多域在策蒸馏 (MOPD - Multi-domain On-Policy Distillation) 样本构造与训练方法**：
    为了在多业务场景联合强化学习对齐（MoE 权重分配）的长链 RL 优化中，防止垂直领域的“基底推荐能力”发生漂移和崩塌，OneReason 提出 MOPD 机制。
    1. **领域教师库（Domain Teachers）**：收集并保存短视频、直播、广告、电商四个垂直场景在上游 CPT / SFT 训练中最优的 Checkpoint 模型作为领域教师 $\pi_{\text{teacher}}^d$ 。
    2. **输入样本池（Multi-domain Prompt Set）**：构建包含各垂直业务场景推荐日志行为的多域 Prompt 集合 $\mathcal{D}_d$ 。
    3. **在策采样与蒸馏（On-Policy Distillation）**：训练时，学生策略模型 $\pi_{\theta}$ 接收来自各域的行为 Prompt $x^d \in \mathcal{D}_d$ ，进行在线的 **On-policy 路径采样**，获得当前的推荐商品 SID 序列 rollouts $y^d \sim \pi_{\theta}$ 。
    4. 将这串由学生自己生成的推荐路径输入其对应的垂直领域教师模型 $\pi_{\text{teacher}}^d$ 。教师模型不需要在全词表上计算完整的条件分布，只需计算该路径下的词元 log 概率（Token-level log-probabilities），用作稠密的纠错指导信号（Dense distillation signal），通过最小化 Reverse KL 散度约束学生模型：

        $$
        \mathcal{L}_{\text{MOPD}}(\theta) = \sum_{d \in \text{Domains}} \mathbb{E}_{x^d \sim \mathcal{D}_d, y^d \sim \pi_{\theta}} \left[ \text{D}_{\text{KL}} \left( \pi_{\theta}(y^d \mid x^d) \parallel \pi_{\text{teacher}}^d(y^d \mid x^d) \right) \right]
        $$

        将该 dense 信号作为辅助正则项，与 Unified RL 目标结合：

        $$
        \mathcal{L}_{\text{Unify}} = \sum_{d \in \text{Domains}} \mathcal{L}_{\text{RL-domain-d}}(\theta_{\text{unified}}) + \mu \cdot \mathcal{L}_{\text{MOPD}}(\theta_{\text{unified}}) + \nu \cdot \mathcal{L}_{\text{General-Reasoning}}(\theta_{\text{unified}})
        $$

---

## 三、核心讨论与对比分析

### 3.1 样本构造方法：数据驱动 (Data-driven) vs 模型生成 (Model-generated)

快手论文中对于 SFT 和 RL 样本的来源经历了两次路线斗争：

| 维度 | 数据驱动 (基于用户真实 Log) | 模型生成 (基于大模型/奖励模型模拟) |
| :--- | :--- | :--- |
| **代表机制** | OneSearch (点击对), OpenOneRec (用户日志) | OneRec (IPA 模拟 Negatives), OneSearch-V2 (LLM 生成 CoT) |
| **优点** | 行为信号绝对真实，能够直接反映用户最终转化（CTR/CVR）。 | 可以生成丰富的反事实（Counterfactual）样本，解决搜推只有正反馈、缺乏负反馈的稀疏性问题。 |
| **缺点** | 只有曝光过的 Item 有 Log，马太效应严重，对于未曝光的冷启动和长尾物料极度不友好。 | 严重依赖 Reward Model 的打分精度，一旦 Reward Model 发生“漂移”，会导致 Policy 崩溃。 |
| **快手的选择** | **混合制 (Hybrid)**：SFT 阶段主要由真实用户点击流（数据驱动）主导；RL/DPO 阶段通过 Reward Model 和 LLM 生成（模型生成）偏好配对。 |

---

### 3.2 RL 与 OPD 阶段训练样本与 SFT 样本的本质区别与构造方法

这是一个非常关键的设计问题：**RL（强化学习）和 OPD（在策蒸馏）阶段所用的训练样本并不是简单的 SFT 样本，在样本结构、目标标签来源和生成机制上有着本质的区别。**

#### 1. SFT 样本 vs RL 样本 (以 DPO/GRPO 为例)

| 维度 | SFT (监督微调) 样本 | RL (强化学习偏好对齐) 样本 |
| :--- | :--- | :--- |
| **样本构成** | 单个 Prompt 与单条正反馈（Clicked/Bought）Target 对： $(x_u, y_{\text{true}})$ 。 | 偏好对 $(x_u, y_w, y_l)$ （DPO）或多路采样轨迹组 $\{y_1, y_2, \dots, y_G\}$ （GRPO）。 |
| **Prompt 部分 ($x_u$)** | 用户画像 + 行为历史序列 + 任务指令。 | 与 SFT 阶段的 Prompt 结构基本一致（复用 SFT 数据集的输入端）。 |
| **Target 部分 ($y$)** | **静态的 Ground-Truth**：历史日志中用户真实点击或购买的商品离散 SID。 | **动态的模型自生成轨迹**：在线或离线通过当前 Policy 模型预测输出的候选推荐序列。 |
| **构造方法与标签来源** | 直接读取离线用户交互日志（数据驱动）。 | **在线在策生成 (On-Policy Rollout)**：<br>1. 输入 $x_u$ 后，利用 Policy 通过 Beam Search 或随机采样（如 top-p）自主生成多条候选路径。<br>2. 用 Reward Model（三塔或多维奖励模型）对自生成序列进行打分。<br>3. 在 DPO 中选出得分最高（Chosen $y_w$ ）与最低（Rejected $y_l$ ）的对；在 GRPO 中直接以这组候选的奖励相对优势值作为梯度更新权重。 |

> [!IMPORTANT]
> **RL 阶段的本质区别在于它引入了负反馈（Rejected 样本）或相对比较。** SFT 数据只告诉模型“用户喜欢什么”，而 RL 样本告诉模型“在模型自己生成的多种推荐结果中，哪一个比另一个更好”，从而消除了曝光偏差，并提供纠错梯度。

#### 2. SFT 样本 vs OPD/MOPD 样本 (在策蒸馏)

| 维度 | SFT 样本 | OPD / MOPD 样本 |
| :--- | :--- | :--- |
| **样本构成** | 具体的、静态的离散 Token 序列对： $(x_u, y_{\text{true}})$ 。 | 只利用 Prompt 集合 $\mathcal{D}_{\text{prompts}}$ ，没有静态的目标 Target 文本。 |
| **Prompt 部分 ($x_u$)** | 纯推荐行为序列。 | **多任务混合 Prompt 集合**：包含推荐行为 Prompt，以及为了防止忘却而混合的**通用自然语言/推理（MMLU, Commonsense QA）Prompt**。 |
| **Target 标签来源** | 历史真实交互的商品 SID。 | **教师模型（Teacher）的软概率分布**：<br>1. 学生模型 $\pi_{\theta}$ 在策自生成推荐轨迹 $y \sim \pi_{\theta}$ 。<br>2. 将该轨迹在教师模型 $\pi_{\phi}$ 上的词表预测概率分布（Soft Probability Distribution）作为目标标签。<br>3. 通过最小化 Reverse KL 散度，强迫学生模型在自己走过的路径上向教师的概率对齐。 |

> [!NOTE]
> **OPD 的本质是在策知识巩固，不需要显式的 Ground-truth 样本标签。** 它通过使用混合 Prompt 库作为输入，在训练中实时计算“学生输出”与“教师输出”的距离，确保模型在学习推荐对齐时，通用语义和推理链不会发生崩塌。

---

### 3.3 损失函数与对齐算法的演进

快手搜推团队对经典强化学习算法（DPO/PPO/GRPO）进行了场景化的改造：

```text
对齐算法演进路径：
1. 经典 DPO (OneRec V1) ──► 2. GBPO 限制剪切 (OneRec V2) ──► 3. TPMA-GRPO 令牌位置分配 (OneSearch-V2)
```

1.  **经典 DPO 局限**：在推荐中对噪声极其敏感，容易因低质负样本导致推荐流质量崩溃。
2.  **GBPO 改良**：引入梯度边界，即使面对极端的正负样本概率比率，也不会丢弃负样本的更新梯度，维持了模型的纠错能力。
3.  **TPMA-GRPO 突破**：引入前缀门控和边际优势计算。解决了生成式搜推中“生成长序列 SID 时无法进行 Token 级信用归因”的本质难题。

---

## 四、快手对齐技术演化趋势与关键结论

通过对 7 篇论文的剖析，可以得出以下关于生成式搜推 CPT/SFT/RL 的核心演化趋势：

### 1. 训练重心的转移：SFT 占比下降，CPT (Perception) 与 RL 成为核心杠杆
在早期（OneSearch, OneRec V1），团队把大量精力放在设计复杂的“多阶段 SFT”上。但到了 OneReason 阶段，团队发现**单纯依靠 SFT 灌输推荐知识是有上限的，必须依靠大规模的 CPT 感知预训练和 RL 对齐来激活模型的隐式推理能力**。

### 2. 推理路径 (CoT) 的“虚”与“实”
*   **OneRec-Think 的局限**：纯粹堆砌 Item 符号的位置 CoT 是“虚”的，模型并未真正建立起物料和现实语义的联结。
*   **OneReason 的纠偏**：设计了结构极其严密的“感知-认知双重增强三级 CoT”。只有将离散的 SID Token 在 CPT 阶段强力接地到自然语言语义空间，后期的 CoT 推理才是“实”的，才能真正发挥 LLM 的世界知识优势。

### 3. 工业落地的“延迟屏障”与架构解耦
在学术界，直接生成数百个 CoT Token 是可行的；但在工业界（如快手这样处理 QPS 达数十万的系统），这不可接受。
*   **架构折中方案**：从 OneRec-Think 的 **Think-Ahead**（前置计算 thought 并缓存）到 OneSearch-V2 的 **Reasoning-Internalized Self-Distillation**（将推理能力完全蒸馏进模型权重），这表明**工业落地的终极方案是“内化推理”，在线不输出中间 CoT 步骤**。

---

## 五、样本格式设计与对齐算法对照表

| 阶段 | 论文 | 核心样本构造方法 | 输入/输出格式设计 (Format) | 核心算法与 Loss 设计 |
| :--- | :--- | :--- | :--- | :--- |
| **CPT** | **OpenOneRec** | Itemic-Text 语义共现，混合通用文本与行为序列 | `[SID Token] <-> [Detailed Item Captions]` 双向预测 | 统一因果语言建模 (CLM) Loss，比例 7:2:1 |
| **CPT** | **OneReason** | Token/Item/Relation/User 四粒度渐进感知对齐 | 关系预测、用户画像建模等多层级 Prompt | CLM Loss + 长上下文 (32K) 序列对齐 |
| **SFT** | **OneSearch** | 三阶段课程学习样本 (语义->共现->个性化) | `User Profile | History | Query -> Output Target SID` | 带自适应奖励加权的 Cross-Entropy Loss |
| **SFT** | **OneRec-Think** | 包含 Reflection/Induction/Deduction 的 CoT 样本 | `<thought> Reasoning step-by-step </thought> Recommend: [SID]` | 只对 Thought 和 Target 算 Loss，Mask 掉 Input |
| **SFT** | **OneReason** | 认知增强三级 CoT (Item-Relation, Abstraction, Synthesis) | 多模态输入下的结构化推理链 `<thought> ... </thought> Target: [SID]` | 强化 Item 感知约束下的 Sequence-to-Sequence Loss |
| **RL** | **OneRec** | 基于 Reward Model 采样构造 Self-Hard 偏好对 | `Winner Sequence vs Loser Sequence` 偏好对 | 列表级直接偏好优化 (Listwise DPO) |
| **RL** | **OneRec-V2** | 对数时长分桶样本，GBPO 梯度限制优化 | 排除时长干扰后的标准化观看时间偏好对 | 梯度有界策略优化 (GBPO) + 时长奖励塑形 |
| **RL** | **OneSearch-V2** | TPMA Token 级位置边际优势，多目标加权 | Group 级 Candidate 序列与多维度反馈指标 | TPMA-GRPO 列表奖励 + 前缀正确性门控梯度 |
| **RL** | **OneReason** | 多业务垂类强化对齐样本 (Specialize-then-Unify) | 垂类多任务对齐 SFT 混合格式 | 多任务 RL 混合策略优化 + MoE 对齐蒸馏 |
