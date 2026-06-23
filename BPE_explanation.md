# 深度解析：Byte-Pair Encoding (BPE) 算法原理、数学推导与高效工程实现

本篇文档对大语言模型（LLM）中广泛采用的 **BPE (Byte-Pair Encoding, 字节对编码)** 分词算法进行深入剖析。我们将结合板书 `BPE.png` 中的内容，从算法演进历史、数学形式化定义、基于单词频数（Word Counts）的高效算法设计、复杂度推导，以及 GPT/Tiktoken 的工程实现等维度进行全方位解析。

---

## 一、 `BPE.png` 板书内容结构解读

板书 `BPE.png` 梳理了 BPE 算法的核心逻辑及其演进脉络，主要分为以下几个部分：

### 1. 核心主题与演进历史（板书上方区域）
- **核心课题**：**“如何科学高效 in 聚合 Bytes”**。这指出了现代大模型分词器的本质：将最原始的字节序列（Bytes）通过科学、高效的算法聚合为更高级的语义单元（Tokens）。
- **发展历史**：
  - **1994年提出 for 压缩**：BPE 最早由 Philip Gage 于 1994 年在论文 *“A New Algorithm for Data Compression”* 中提出，最初作为一种无损数据压缩算法。
  - **2016年用于 NLP 模型**：Sennrich 等人在论文 *“Neural Machine Translation of Rare Words with Subword Units”* 中首次将 BPE 引入自然语言处理领域，用于解决机器翻译中的未登录词（OOV）与稀有词问题。
  - **2019年用于 GPT-2**：Radford 等人在 GPT-2 中引入了 **Byte-level BPE (BBPE)**，将初始词表大小定为 256 个基础字节，从而实现了无 OOV 的分词设计。

### 2. 高效算法的伪代码逻辑（板书左右两侧区域）
板书给出了一个基于哈希表与优先队列（堆）的高效 BPE 训练伪代码。

**左侧：初始化阶段 (Initialization)**

*   **计算需要合并的次数** `num_merges`（以 $M$ 表示，其中 $|V|$ 为目标词表大小）：

$$
M = |V| - 256
$$

*   **将输入文本使用 UTF-8 编码转化为字节序列，并转化为整数 Token 列表** `tokens`：

```python
tokens = list(data.encode('utf-8'))
```

*   **声明数据结构与初始化**：
    声明两个核心数据结构：`pair_counts`（使用最大堆 Heap 维护，大小约 1 万）和 `pair_positions`（映射到位置列表的指针）。遍历相邻 Token，统计所有相邻 Pair 的频数与其出现的索引位置（`pos`）。

**右侧：合并阶段 (Merge Loop)**

循环执行 $M$ （即 `num_merges`）次合并：

**1. 弹出频数最高字节对**：从最大堆 `pair_counts` 中弹出当前频数最高的相邻字节对 `pair`。

**2. 产生新 Token ID**：产生新的 Token ID（以 $T_{\text{new}}$ 表示）：

$$
T_{\text{new}} = 256 + i
$$

在代码中对应的变量为 `new_token`。

**3. 记录合并规则**：将 `pair` 映射为 `new_token`。

**4. 替换 Token**：遍历该 `pair` 出现的所有位置 `pos`，在 Token 链中将该相邻对替换为 `new_token`。

**5. 局部更新**：修正 Token 链（双向链表），减少旧 Pair 的计数（`pair_counts[old_pair] -= 1`），增加新生成 Pair 的计数（`pair_counts[new_pair] += 1`），并实时修复位置索引表 `pair_positions`。这避免了全局扫描，实现了极高的运算效率。

---

## 二、 BPE 算法的数学建模与理论分析

BPE 算法在本质上可以看作是一种**自底向外的贪心层次聚类算法**，其核心优化目标是在限制词表大小的前提下，最大化语料库的压缩率（或最小化编码长度）。

### 1. 形式化定义

设语料库为字节序列 $S = (s_1, s_2, \dots, s_N)$ ，其中初始字符集（即基础字节）为：

$$
V_0 = \{b_0, b_1, \dots, b_{255}\}
$$

在第 $t$ 次迭代中，我们要从当前的词表 $V_{t-1}$ 中选择两个 Token $u, v \in V_{t-1}$ 进行合并，生成一个新 Token $w_t = u \oplus v$ 。

合并的选择准则为**联合出现频率最大化**：

$$
(u^{\ast}, v^{\ast}) = \arg\max_{(u, v) \in V_{t-1} \times V_{t-1}} \text{Freq}(u, v; S_{t-1})
$$

其中 $\text{Freq}(u, v; S_{t-1})$ 表示在序列 $S_{t-1}$ 中，非重叠 of 相邻对 $(u, v)$ 出现的次数。

合并后，词表与序列分别更新为：

$$
V_t = V_{t-1} \cup \{w_t\}
$$

$$
S_t = \text{Replace}(S_{t-1}, (u^{\ast}, v^{\ast}) \to w_t)
$$

### 2. 信息论与最小描述长度（MDL）视角

我们可以从**最小描述长度 (Minimum Description Length, MDL)** 框架来解释 BPE。MDL 原理认为，最好的模型是能够使模型本身的描述长度与模型对数据的压缩长度之和最小的模型。

对于 BPE，我们希望用最少的总比特数来传输语料库 $S$ 和词表（规则集） $V$ 。

假设语料库 $S_t$ 的长度为 $|S_t|$ ，词表中每个 Token 的平均表示代价为 $L_{\text{tok}}$ 。则传输语料库的代价为：

$$
\text{Cost}(S_t) = |S_t| \cdot L_{\text{tok}}
$$

而传输词表规则（包含合并字典）的代价为：

$$
\text{Cost}(V_t) = |V_t| \cdot L_{\text{rules}}
$$

总代价为：

$$
\text{TotalCost}(t) = |S_t| \cdot L_{\text{tok}} + |V_t| \cdot L_{\text{rules}}
$$

当我们将频数为 $k$ 的 Pair $(u, v)$ 合并为新 Token $w$ 时，会产生以下两个层面的变化：

**第一，语料库的长度减少了 $k$** （因为每次合并将 2 个 Token 缩减为 1 个 Token）：

$$
|S_t| = |S_{t-1}| - k
$$

**第二，词表大小增加了 1**：

$$
|V_t| = |V_{t-1}| + 1
$$

因此，两步迭代之间的总代价变化量为：

$$
\Delta \text{TotalCost} = \text{TotalCost}(t) - \text{TotalCost}(t-1) = -k \cdot L_{\text{tok}} + L_{\text{rules}}
$$

为了使总代价最小化，我们需要在每一步使 $\Delta \text{TotalCost}$ 尽可能为负且绝对值最大。因为 $L_{\text{tok}}$ 和 $L_{\text{rules}}$ 在局部迭代中可近似看作常数，这等价于：

$$
\arg\max_P (-\Delta \text{TotalCost}) \iff \arg\max_P k
$$

这从信息论角度严格证明了**为什么 BPE 贪心选择频数最大的字节对进行合并是合理的**——它在每一步都实现了局部最优 of 无损数据压缩。

---

## 三、 基于唯一单词词频统计（Word Counts）的高效算法

板书虽然给用了基于全局双链表与堆的伪代码实现，但在实际 Python 工程中（例如 GPT-2/GPT-4 词表构建中），**基于唯一单词词频统计（Word Counts）的训练方案**在效率和内存占用上表现得更佳。

### 1. 全局双链表与单词词频的性能差异
- **全局双链表法（理论复杂度 $O(N \log N)$ ）**：
  - 维护整个语料的双向链表。
  - 需要在 Python 中为几千万个字符各自创建 `Node` 实例。这不仅消耗上 GB 的内存，而且因为 Python 的指针追逐、垃圾回收等开销，在纯 Python 环境下运行速度极慢。
- **单词词频法（Word Counts）**：
  - 真实预训练文本中，存在大量的重复词汇（如 `the`，`and` 等）。
  - 我们先对语料分词，得到唯一单词的统计字典 `word_counts`，其大小 $W$ 远小于语料总字数 $N$ （例如，在 10MB 的文本中， $N \approx 10^7$ ，而唯一单词种类数 $W \approx 30000$ ）。
  - 每次合并只需对 `word_counts` 进行遍历。因此单步迭代的开销仅为 $O(W \cdot L_{\text{avg}})$ （ $L_{\text{avg}}$ 是单词平均长度，通常约等于 5）。
  - 该方案极大地利用了 CPython 底层对 `dict` 和 `tuple` 的高度优化，在 Python 环境下运行速度能够达到前者的数十倍以上。

---

### 2. 单词词频 BPE 的具体步骤

1.  **分词与映射**：
    将语料库通过 GPT-2 预分词正则表达式切分，转化为一个字符元组的计数词典：
    `word_counts[('h', 'e', 'l', 'l', 'o')] = count`
2.  **全局统计相邻 Pair**：
    遍历 `word_counts`，将所有单词内的相邻 pair 频数加权求和。
3.  **查找最优 Pair 并合并**：
    检索频数最高的 pair，使用**确定性 tie-breaking** 机制打破平衡：
    如果出现频数相同，则选择**原始字节序列字典序最大**的 pair。
4.  **单词库局部合并**：
    遍历唯一单词列表，在各单词内部，将对应的两相邻字符合并为新的连接字符串。为了提高替换速度，可利用 `word.index(best[0], i)` 方法跳过不包含合并目标的字符，从而大幅度减少无效的遍历开销。
5.  **更新词表与合并表**：
    重复此步骤直到达到设定的 `vocab_size`。

---

### 3. 算法复杂度分析

*   **初始化阶段**：
    *   正则分词与字节-Unicode 映射： $O(N)$ 。
    *   统计唯一单词词频： $O(N)$ 。
*   **训练合并阶段（共 $M$ 次合并，其中 $M = V - 256 - |S_{\text{special}}|$ ）**：
    *   每一步迭代中，扫描不同 pair 数量。在最坏情况下，不同 pair 数量不超过唯一单词的总字符数 $W \cdot L_{\text{avg}}$ 。
    *   计算 pair 频数并检索最大值： $O(W \cdot L_{\text{avg}})$ 。
    *   更新唯一单词字典： $O(W \cdot L_{\text{avg}})$ 。
    *   对于 $M$ 次合并，总的复杂度为：

$$
T_{\text{train}} = O(M \cdot W \cdot L_{\text{avg}})
$$

因为 $W \cdot L_{\text{avg}} \ll N$ ，在文本规模庞大时，该训练效率显著优于 Naive 语料库全局扫描方法。

---

## 四、 字节级 BPE（Byte-level BPE）与 GPT / Tiktoken 实践

### 1. 字节-Unicode 映射的工程意义
在进行字节级 BPE 训练时，必须解决“部分字节在 Unicode 中无法安全表示（如无效的 UTF-8 字节）”的问题。
GPT-2 引入了一个**字节-Unicode 映射（Bytes-to-Unicode）**：
- 它将 256 个基本字节映射到 256 个可以安全显示的、互不重复的 Unicode 字符（排除了控制字符和空格等，防止正则表达式切分时出错）。
- 这不仅彻底解决了 UNK（Out-of-Vocabulary）问题，使任何文本均能被成功分词，也极大方便了用正则表达式进行分词。

---

### 2. GPT 预分词模式的作用
为了防止标点符号、数字和单词在 BPE 合并时产生杂乱的无语义拼接（如 `the.` 或 `12,`），GPT-2 规定：文本首先由特定的正则表达式切分，而后**合并绝对只在切分出来的子串内发生**。

#### GPT-2 预分词正则表达式：
```python
import regex as re
GPT2_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
```
该正则优先提取带有可选前导空格的纯字母段、纯数字段、非空白的标点及符号段、空格段等，起到了天然的词汇边界屏障作用。

---

### 3. LRU Cache 分词加速优化
在推理阶段（Encoding），对于同一个单词，分词器的输出（即它被切分成哪些 subwords）是完全确定的。
由于大部分文本中高频词重复度极高，在 `BPE.encode` 中引入 **LRU 缓存（Least Recently Used Cache）**：
- 缓存键为 Unicode 单词字符串，值为经过 BPE 合并后的 subword 元组。
- 在实际评测中，加入 LRU 缓存后，BPE Tokenizer 的编码速度通常能够提升 5 到 10 倍，也是 `tiktoken` 能够实现数十万 Token/秒级别吞吐的关键一环。

---

## 五、 高效 BPE 接口与工程实现细节

在我们的高效实现中，为满足模块化和流式编码需求，核心包含以下几个主要接口：

*   **`encode(self, text: str) -> list[int]`**：接收纯文本，进行最长匹配特殊 Token 切分、正则预分词以及 LRU 缓存加速的 BPE 编码。
*   **`decode(self, ids: list[int]) -> str`**：接收 Token 列表，解码回对应的字节流，并使用 `errors="replace"` 安全地恢复成 UTF-8 字符串。
*   **`encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]`**：采用生成器流式设计，可以方便处理超大语料（如大文件或在线流式传输），避免了将整文本一次读入内存。
*   **`train_bpe(input_path: str, vocab_size: int, special_tokens: list[str])`**：基于词频方案，融合了 `.index()` 子串快速跳转合并优化和确定性字典序 Tie-Breaking。

---

## 六、 运行示例与代码行为解析

为了更直观地理解 BPE 代码的底层运行逻辑，我们以一个具体的简单文本为例，完整跟踪其训练和推理的内部数据流动。

### 1. 训练阶段跟踪 (BPE Training Walkthrough)

假设我们有一篇极其简单的输入文本数据，内容为：
```text
ababab
```
我们希望通过调用 `train_bpe(input_path, vocab_size=258, special_tokens=["<pad>"])` 训练一个 BPE 分词器。

**初始化阶段**：
*   **基础词表初始化**：
    词表首先初始化为 256 个单字节 Token，映射为对应的单字节。
    `vocab[0] = b"\x00"`, ..., `vocab[255] = b"\xff"`。
*   **特殊 Token 插入**：
    将特殊 Token `"<pad>"` 插入词表，分配 ID 为 256。此时词表大小为 257：

$$
|V| = 257
$$

*   **预分词与唯一单词统计**：
    输入文本为 `"ababab"`，经过 `GPT2_SPLIT_PATTERN` 正则分词得到单元素列表 `["ababab"]`。
    使用 `bytes_to_unicode` 将每个字节映射到安全 Unicode。对于字母 `'a'` 和 `'b'`，它们会映射为自身。
    统计唯一单词词频，得到词频字典 `word_counts`：
    `word_counts[('a', 'b', 'a', 'b', 'a', 'b')] = 1`。

**循环合并阶段（当 $|V| < 258$ 时持续迭代）**：
由于我们的目标 `vocab_size = 258`，当前词表大小为 257，因此需要执行 1 次合并。

**迭代 1**：
*   **统计相邻 Pair 的频数**：
    遍历 `word_counts` 中的所有相邻 Token 对：
    - `('a', 'b')` 在单词内相邻出现了 3 次，加权频数为 $3 \times 1 = 3$ 。
    - `('b', 'a')` 在单词内相邻出现了 2 次，加权频数为 $2 \times 1 = 2$ 。
*   **选择最优 Pair**：
    检索到频数最大的 pair 为 `('a', 'b')` 。
*   **合并并更新词表**：
    生成新 Token 字符串 `'ab'` ，将其对应的原始字节序列 `b"ab"` 存入词表，分配新 ID 为 257。同时将 `(b"a", b"b")` 记录到合并规则 `merges` 中。此时词表大小达到 258：

$$
|V| = 258
$$

*   **更新 word_counts**：
    遍历 `word_counts` 的键，利用快速的 `.index()` 扫描将相邻的 `'a'` 和 `'b'` 替换为合并后的 `'ab'`。
    原元组 `('a', 'b', 'a', 'b', 'a', 'b')` 被局部合并为 `('ab', 'ab', 'ab')`。
    更新后 `word_counts` 变为：
    `word_counts[('ab', 'ab', 'ab')] = 1`。

词表大小已达到 258，训练循环结束。返回 `vocab` 和 `merges`。

---

### 2. 推理阶段编码跟踪 (BPE Encoding Walkthrough)

现在我们使用刚刚训练好的词表，调用 `BPE.encode("ababab")` 进行编码。

**分词与字节映射**：
*   输入 `"ababab"`，正则切分为 `["ababab"]`。
*   转换为安全 Unicode 表示为：`'a'`, `'b'`, `'a'`, `'b'`, `'a'`, `'b'`。
*   调用 `self.bpe('ababab')` 进行子串级合并。

**BPE 贪心合并过程**：
*   **初始状态**：
    `word = ('a', 'b', 'a', 'b', 'a', 'b')`
    `pairs = {('a', 'b'), ('b', 'a')}`
*   **合并检索**：
    我们在合并优先级排行 `self.bpe_ranks` 中寻找 `pairs` 中排名最高的项。
    `self.bpe_ranks` 中仅有 `('a', 'b')` 对应的级别为 0。
    选择最优 pair `('a', 'b')` 。
*   **局部合并**：
    利用 `.index()` 扫描，将相邻的 `'a'` and `'b'` 替换为 `'ab'`。
    合并后的 `word` 变为 `('ab', 'ab', 'ab')`。
*   **再次检索**：
    新的相邻 `pairs` 仅为 `{('ab', 'ab')}`，由于其不在合并映射 `self.bpe_ranks` 中，合并循环 break 结束。
*   **最终返回**：
    BPE 编码函数返回元组 `('ab', 'ab', 'ab')`。

**映射为 ID 列表**：
*   遍历返回的 subword 元组，将每个部分映射回字节流：
    - `'ab'` 映射回 `b"ab"`。
*   在 `self.encoder` 中查询对应的 ID，`b"ab"` 对应的 ID 为 257。
*   最终返回的 ID 编码序列为 `[257, 257, 257]`。

