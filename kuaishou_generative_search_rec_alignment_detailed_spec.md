# 快手生成式搜推论文 CPT / SFT / RL 阶段样本构造格式与方法完整规范

本规范旨在不进行总结提炼的前提下，完整罗列快手生成式搜推系列 7 篇论文中，在持续预训练（CPT）、监督微调（SFT）和强化学习偏好对齐（RL）阶段所采用的样本构造格式、Prompt 模板以及数学公式。

---

## 1. OneSearch

### 1.1 CPT / 语义对齐阶段 (Semantic Alignment)
通过双向文本重构建立离散语义 ID (SID) 与真实文本的对齐。

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

### 1.2 SFT 阶段 (Co-occurrence & Personalization)
*   **共现对齐格式 (Query-to-Item / Item-to-Query)**：
    ```text
    Input: "Search query: [User Query] -> Predict target item semantic ID:"
    Output: "<sid_level_1_x><sid_level_2_y><sid_level_3_z>"
    ```
    ```text
    Input: "Retrieve search query for item: <sid_level_1_x><sid_level_2_y><sid_level_3_z>"
    Output: "[User Query]"
    ```
*   **用户个性化建模格式**：
    输入端包含用户画像特征、历史购买行为、短期搜索点击行为序列（通过滑动窗口进行切片增强）。
    ```text
    Input: "User Info - Age: [Age], Gender: [Gender], City: [City] | Long-term Pref: [Brand_A, Category_B] | Short-term Behaviors: [Query_1 -> Click_Item_1_SID, Query_2 -> Click_Item_2_SID] | Current Query: [Current Query]"
    Output: "Target Item SID: <sid_level_1_x><sid_level_2_y><sid_level_3_z>"
    ```

### 1.3 RL / 偏好对齐阶段 (PARS)
*   **自适应奖励权重设计**：
    根据交互行为深度设计 6 级基础奖励权重 $\lambda$：
    $$\lambda = [\lambda_{\text{buy}}, \lambda_{\text{add\_to\_cart}}, \lambda_{\text{collect}}, \lambda_{\text{click}}, \lambda_{\text{expose\_unclick}}, \lambda_{\text{unexpose}}]$$
    具体配置为：$\lambda = [2.0, 1.5, 1.0, 0.5, 0.2, 0.0]$。
*   **多塔校准奖励分数 $r(q, i)$**：
    $$r(q, i) = w_{\text{ctr}} \cdot P(\text{click} \mid q, i) + w_{\text{cvr}} \cdot P(\text{buy} \mid \text{click}, q, i)$$
    其中，三塔结构分别预测 CTR、CVR 和 CTCVR。
*   **Listwise 排序优化 Loss 函数**：
    $$\mathcal{L}_{\text{PARS}} = -\sum_{q} \log \frac{\sum_{i \in \mathcal{I}^+} \exp(r(q, i) / \tau)}{\sum_{j \in \mathcal{I}^+ \cup \mathcal{I}^-} \exp(r(q, j) / \tau)}$$
    $\mathcal{I}^+$ 代表正反馈商品集合，$\mathcal{I}^-$ 代表曝光未点击等负反馈商品集合，$\tau$ 为温度超参数。

---

## 2. OneSearch-V2

### 2.1 SFT 阶段 (Thought-Augmented Query Understanding)
引入显式 CoT 推理链样本，在模型输出 SID 之前强制生成 `<thought>` 标签内容。
*   **Thought-Augmented 格式**：
    ```text
    Prompt: User Profile: [Profile] | Current Query: [Complex/Ambiguous Query]
    Response: <thought>
    - Query intent parsing: The user is looking for [intent].
    - Preference constraints: The user prefers [brand/category/price-range].
    - Deduction: Recommend items matching both the intent and historical preference.
    </thought>
    Target Item SID: <sid_level_1_x><sid_level_2_y><sid_level_3_z>"
    ```

### 2.2 RL 阶段 (TPMA-GRPO)
*   **TPMA-GRPO 优化目标函数**：
    $$\mathcal{L}_{\text{TPMA}} = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{L} \sum_{t=1}^{L} \text{gate}_{i,t} \cdot \min \left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right)$$
    其中 $G$ 为 Rollout 样本数量，$L$ 为序列长度，$r_{i,t}(\theta)$ 为当前策略与基准策略的概率比值：
    $$r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} \mid x_u, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid x_u, o_{i,<t})}$$
*   **前缀正确性门控机制 (Prefix Gate)**：
    $$\text{gate}_{i,t} = \prod_{j=1}^{t-1} \mathbb{I}(o_{i,j} == o_{i,j}^*)$$
    其中 $\mathbb{I}$ 是指示函数，$o_{i,j}^*$ 代表真实的 Ground-Truth Token。如果前 $t-1$ 个 Token 中存在错误，则第 $t$ 步的梯度被完全屏蔽（$\text{gate}_{i,t} = 0$）。
*   **Token 位置级边际优势 $\hat{A}_{i,t}$**：
    $$\hat{A}_{i,t} = R_{i, \ge t} - \bar{R}_{\ge t}$$
    其中 $R_{i, \ge t}$ 是第 $i$ 个轨迹从位置 $t$ 开始的累积奖励，$\bar{R}_{\ge t}$ 是当前组内所有轨迹在位置 $t$ 开始的平均奖励。

---

## 3. OneRec

### 3.1 CPT 阶段 (Multimodal & Collaborative Alignment)
*   **多模态 Caption 生成任务 (Caption-Gen) 格式**：
    ```text
    Input: "Generate a dense description for the video frame sequence [Frames] and audio transcript [ASR]:"
    Output: "[Detailed Video Caption]"
    ```
*   **I2I 协同对齐任务格式**：
    基于用户共同点击行为（如 Swing 相似度高），构造对比样本对。
    ```text
    Input: "Anchor Item: <sid_level_1_a><sid_level_2_b><sid_level_3_c> | Match similar item:"
    Output: "<sid_level_1_x><sid_level_2_y><sid_level_3_z>"
    ```

### 3.2 SFT 阶段 (Session-Wise Generation)
推荐任务被建模为会话级序列生成，使用 NTP (Next Token Prediction) 对 Target Sequence 的每一个 SID Token 算 Cross-Entropy Loss。
*   **Session-Wise NTP 格式**：
    ```text
    Prompt: User history sequence: [SID_1, SID_2, ..., SID_n] | Context: [Time_of_Day, Device]
    Response: [SID_n+1, SID_n+2, ..., SID_n+k]
    ```
    计算 Loss 时对 Prompt 部分的 Token 进行 Mask：
    $$\mathcal{L}_{\text{NTP}} = -\sum_{t=n+1}^{n+k} \log P(SID_t \mid SID_{<t}, \text{Prompt})$$

### 3.3 RL / DPO 阶段 (Iterative Preference Alignment)
*   **基于 Beam Search 的 Chosen/Rejected 样本构造**：
    1. 给定 Prompt $x_u$ ，当前 Policy $\pi_{\theta}$ 通过 Beam Search 生成 $M$ 个候选推荐序列 $\{q_1, q_2, ..., q_M\}$。
    2. 使用多目标奖励模型 $R(x_u, q_j)$ 计算偏好得分。
    3. 选择得分最高和最低的序列对构成 $(q_w, q_l)$：
       $$q_w = \arg\max_{q_j} R(x_u, q_j), \quad q_l = \arg\min_{q_j} R(x_u, q_j)$$
*   **DPO Loss 函数**：
    $$\mathcal{L}_{\text{DPO}}(\theta; \theta_{\text{ref}}) = -\mathbb{E}_{(x_u, q_w, q_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(q_w \mid x_u)}{\pi_{\text{ref}}(q_w \mid x_u)} - \beta \log \frac{\pi_{\theta}(q_l \mid x_u)}{\pi_{\text{ref}}(q_l \mid x_u)} \right) \right]$$

---

## 4. OneRec-V2

### 4.1 SFT 阶段 (Loss Average Adaptation)
*   **Loss 计算机制调整**：
    从 Sum 改为 Mean，计算 3 层 SID Token 的平均交叉熵。
    $$\mathcal{L}_{\text{SFT\_Mean}} = -\frac{1}{3} \sum_{k=1}^{3} \log P(s_k \mid s_{<k}, x_u)$$

### 4.2 RL 阶段 (GBPO & Duration Shaping)
*   **对数时长分桶函数**：
    $$F(d) = \lfloor \log_{\beta}(d + \epsilon) \rfloor$$
    $\beta$ 为对数底数，$\epsilon$ 为平滑常数，$d$ 为视频原始播放时长（Duration）。
*   **时间感知标准化观看时长奖励**：
    $$R_{\text{duration}}(pt, d) = \frac{pt - \mu_{F(d)}}{\sigma_{F(d)}}$$
    其中 $pt$ 为用户实际观看时间，$\mu_{F(d)}$ 和 $\sigma_{F(d)}$ 是时长分桶 $F(d)$ 内部所有样本观看时间的均值与标准差。
*   **GBPO (Gradient-Bounded Policy Optimization) 剪切比率公式**：
    引入参考梯度边界 $B(x_u, q_w, q_l)$，对策略概率比进行动态控制：
    $$\mathcal{L}_{\text{GBPO}}(\theta) = -\mathbb{E} \left[ \min \left( \hat{r} \hat{A}, \text{clip}(\hat{r}, 1 - B(\theta_{\text{old}}), 1 + B(\theta_{\text{old}})) \hat{A} \right) \right]$$
    其中 $\hat{r} = \frac{\pi_{\theta}(q_w \mid x_u) / \pi_{\theta}(q_l \mid x_u)}{\pi_{\theta_{\text{old}}}(q_w \mid x_u) / \pi_{\theta_{\text{old}}}(q_l \mid x_u)}$，梯度边界定义为：
    $$B(\theta_{\text{old}}) = \alpha \cdot \left\| \nabla_{\theta_{\text{old}}} \log \frac{\pi_{\theta_{\text{old}}}(q_w \mid x_u)}{\pi_{\theta_{\text{old}}}(q_l \mid x_u)} \right\|_2$$

---

## 5. OneRec-Think

### 5.1 SFT 阶段 (Reasoning Scaffolding)
通过“推理脚手架”显式激活 LLM 在推荐时的分析过程。
*   **Reasoning Scaffolding 格式模板**：
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

### 5.2 RL 阶段 (Multi-Validity Reward Function)
考虑用户兴趣多有效性（即存在多个合理的推荐结果），定义序列级奖励分数：
$$R_{\text{multi}}(Y) = \frac{1}{|Y|} \sum_{y \in Y} \max_{j \in \mathcal{I}^+} \text{Similarity}(y, j)$$
$\text{Similarity}(y, j)$ 为预测推荐 ID $y$ 与用户真实正反馈 ID $j$ 在 Latent 语义空间中的余弦相似度。

---

## 6. OpenOneRec

### 6.1 Co-Pretraining 阶段
*   **混合配比**：70% 推荐行为序列 + 20% 密集多模态文本描述（Dense Captions） + 10% 通用自然语言文本。
*   **对齐格式**：
    ```text
    Input: "Describe the item with semantic ID <sid_level_1_x><sid_level_2_y><sid_level_3_z> in detail:"
    Output: "Item Title: [Title]. Category: [Category]. Modality Description: A video showing [Content], with tags [Tags]."
    ```

### 6.2 SFT 阶段 (RecIF-Bench 8大任务 Prompt 格式)
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

### 6.3 Post-Training RL / 教师蒸馏阶段
*   **Reverse KL 散度约束优化**：
    $$\mathcal{L}_{\text{蒸馏}}(\theta) = \mathcal{L}_{\text{RL\_Task}}(\theta) + \gamma \cdot \text{D}_{\text{KL}}(\pi_{\text{teacher}} \parallel \pi_{\theta})$$
    其中 $\pi_{\text{teacher}}$ 是通用 LLM（如 Qwen-3-8B-Instruct）在通用自然语言任务上的概率分布，$\pi_{\theta}$ 是正在对齐的推荐模型。

---

## 7. OneReason

### 7.1 CPT 阶段 (Four-Granularity Progressive Perception)
*   **Token Level (Token 粒度) 格式**：
    对 SID Token 本身进行语义分拆和组合预测。
    ```text
    Prompt: Decompose the compound token <sid_level_1_x><sid_level_2_y> into parent category representation:
    Response: Parent Node Category: [Outdoors]
    ```
*   **Item Level (物料粒度) 格式**：
    多维度物料 Q&A。
    ```text
    Prompt: Answer based on item token <sid_level_1_x><sid_level_2_y><sid_level_3_z>: What is the brand and utility of this item?
    Response: The brand is Brand_A, and it is a professional tent used for summer camping.
    ```
*   **Relational Level (关系粒度) 格式**：
    解释协同过滤关系的转移逻辑。
    ```text
    Prompt: Explain the relationship between Item <Item_A_SID> and Item <Item_B_SID>:
    Response: Users who purchased the Item_A_Name (sleeping bag) also frequently co-purchased Item_B_Name (camping tent) for outdoor activities.
    ```
*   **User Level (用户意图粒度) 格式**：
    混合排序与行为序列的总结。
    ```text
    Prompt: Analyze and summarize the core interests from this chronological item sequence: [<Item_1_SID>, <Item_2_SID>]
    Response: Latent User Interest: [Outdoor Sports / Lightweight Camping]
    ```

### 7.2 SFT 阶段 (Cognition-Enhanced CoT)
SFT 数据集包含极其严格的“三级认知增强 CoT”结构，具体包括物料关系归纳、兴趣抽象以及决策合成三个明确段落。
*   **三级认知增强 SFT 样本格式**：
    ```text
    Prompt: User history: [<Item_1_SID>(Category: Running Shoes), <Item_2_SID>(Category: Sports Watch)]. Predict next click.
    Response:
    <thought>
    - Item Relation Induction:
      The user interacted with Item_1 (Running Shoes) and Item_2 (Sports Watch). These items exhibit strong co-occurrence in the "Professional Running/Marathon Preparation" sub-domain.
    - Interest Abstraction:
      The user's latent interest is transitioning from general fitness to structured running training, requiring professional monitoring and gear support.
    - Decision Synthesis:
      Given the user's focus on marathon gears and sports brand loyalty, a high-performance running hydration vest from the same ecosystem is the most logical choice.
    </thought>
    Target: <Item_target_SID>
    ```

### 7.3 RL 阶段 (Specialize-then-Unify)
*   **第一步：垂类专精对齐 (Specialize Phase)**
    在电商、短视频、直播、本地生活广告 4 个场景下，利用特定奖励模型（如 $R_{\text{e-commerce\_cvr}}$ , $R_{\text{video\_watchtime}}$）分别训练专精策略：
    $$\theta_{\text{domain}} = \arg\max_{\theta} \mathbb{E} [ R_{\text{domain}}(x_u, y) ] - \beta \text{D}_{\text{KL}}(\pi_{\theta} \parallel \pi_{\text{sft}})$$
*   **第二步：混合专家网络统一融合 (Unify Phase)**
    通过权重路由机制（Routing Network）融合为统一模型 $\theta_{\text{unified}}$，使用通用认知任务样本进行正则化约束，防止推理逻辑退化：
    $$\mathcal{L}_{\text{Unify}} = \sum_{d \in \text{Domains}} \mathcal{L}_{\text{RL\_domain\_d}}(\theta_{\text{unified}}) + \mu \cdot \mathcal{L}_{\text{General\_Reasoning}}(\theta_{\text{unified}})$$
