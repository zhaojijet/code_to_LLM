import regex as re
from typing import Iterable
import functools

# GPT-2 预分词正则表达式模式
# 该正则主要用于将输入的纯文本流分割成多个子词单元候选（单词、标点、数字、连续空格或缩写等）
# \p{L}+ 匹配连续的字母，\p{N}+ 匹配连续的数字，[^\s\p{L}\p{N}]+ 匹配连续的非空白、非字母、非数字字符（如标点）
GPT2_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

@functools.lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """
    建立单字节（0~255）到安全 Unicode 字符（字符串）的双射映射映射表。
    
    GPT-2 中的这种映射设计的工程目的是：
    1. 确保所有 256 个字节值都能被某个可打印、合法的 Unicode 字符表示，从而消除 UNK (Unknown Token) 的情况。
    2. 避开控制字符（如 0~31, 127 等）和空格等，防止对正则表达式的切分和文本展示带来异常行为。
    """
    # 挑选出标准的可打印、不易冲突的字节区间：
    # 33 ('!') ~ 126 ('~'), 161 ('¡') ~ 172 ('¬'), 174 ('®') ~ 255 ('ÿ')
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # 对剩余不在 bs 中的单字节值（如控制字符和空格），顺次映射到从 256 (2**8) 开始的非冲突 Unicode 码位
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    # 返回 {字节整型值: 映射后的字符} 字典
    return {b: chr(c) for b, c in zip(bs, cs)}

class BPE:
    """
    Byte-Pair Encoding (BPE) 分词推理器类，支持 GPT-2 兼容的字节级分词、特殊 Token 匹配以及缓存加速。
    """
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        初始化 BPE 分词器。
        
        Args:
            vocab: 词表字典，映射 Token ID -> 原始字节串 bytes
            merges: 训练好的字节合并规则列表，其中每个元素为 tuple[bytes, bytes]，指明合并的左右部分
            special_tokens: 特殊 Token 字符串列表（如 ["<pad>", "<unk>"]）
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # 统计特殊 Token 的字符串与其对应 ID 的映射，便于编码时快速检索
        self.special_token_ids = {
            token: idx
            for idx, token_bytes in vocab.items()
            if (token := token_bytes.decode("utf-8", errors="ignore"))
            in self.special_tokens
        }
        
        # 编码表反向查找映射：原始字节 bytes -> Token ID
        self.encoder = {v: k for k, v in vocab.items()}
        
        # 载入字节-Unicode安全映射
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # 构建 BPE 合并优先级排行字典 self.bpe_ranks
        # 合并规则是以 unicode 字符对的形式做匹配，故此处需将 merges 的原始字节元组利用 byte_encoder 转化为对应的 unicode 字符元组
        self.bpe_ranks = {}
        for i, (b1, b2) in enumerate(merges):
            s1 = "".join(self.byte_encoder[b] for b in b1)
            s2 = "".join(self.byte_encoder[b] for b in b2)
            self.bpe_ranks[(s1, s2)] = i
            
        # 词汇缓存：缓存每个单词字符串被切分为 subword 元组的最终结果，极大加速推理阶段编码
        self.cache = {}
        
    def bytes_to_unicode(self):
        """返回当前实例中关联的字节-Unicode 映射字典"""
        return bytes_to_unicode()
        
    def get_pairs(self, word):
        """
        获取一个单词元组（由多字符或合并子词组成）中所有相邻的字符/子词对。
        
        例如：('h', 'e', 'l', 'l', 'o') -> {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')}
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        """
        对单个预分词出来的 token（已被映射为安全 Unicode）执行 BPE 贪心合并。
        
        Args:
            token: 传入的单个 token 字符串（如 'hello'）
        Returns:
            完成合并后的 subword 字符串元组（如 ('hel', 'lo')）
        """
        if token in self.cache:
            return self.cache[token]

        # word 最初是由单个 unicode 字符组成的元组
        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        # 按照训练好的合并优先级（bpe_ranks 中的优先级索引越小越优先）循环合并相邻字符对
        while True:
            # 找到当前相邻 pairs 中优先级最高（即 rank 索引值最小）的 bigram 对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            
            # 若所有相邻 bigram 均未出现在 merges 合并规则中，说明已无可合并内容，退出循环
            if bigram not in self.bpe_ranks:
                break
                
            first, second = bigram
            new_word = []
            i = 0
            # 遍历当前的 word 符号序列，利用 .index() 方法进行快速跨越替换
            while i < len(word):
                try:
                    # 快速跳转定位到第一个合并项的索引位置
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    # 如果后续不存在该合并项，则直接拼接剩余的所有子词并退出
                    new_word.extend(word[i:])
                    break

                # 确认其与后一个元素组合是否完全符合待合并 bigram
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
                    
            word = tuple(new_word)
            # 如果整个单词已经完全合并为 1 个 Token，则停止合并
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
                
        # 写入缓存并返回
        self.cache[token] = word
        return word
        
    def encode(self, text: str) -> list[int]:
        """
        对文本进行 BPE 编码。
        
        1. 使用特殊 Token 对文本进行切分（优先匹配长度更长的特殊 Token）；
        2. 对普通文本区间利用 GPT2_SPLIT_PATTERN 预分词；
        3. 对每个预分词的 Token，通过字节-Unicode 映射并调用 BPE 贪心合并算法将其切分为 subwords；
        4. 将最终切分出的 subwords/特殊 Token 映射成词表对应的整型 Token ID 列表。
        """
        bpe_tokens = []
        # 按长度降序对特殊 Token 进行排序，确保最长匹配优先
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        if sorted_special_tokens:
            # 生成带捕获括号的正则模式，在分割时会同时返回分隔符（即特殊 Token）
            pattern = "(" + "|".join(re.escape(k) for k in sorted_special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                # 特殊 Token 编码处理
                encoded_part = part.encode("utf-8")
                if encoded_part in self.encoder:
                    bpe_tokens.append(self.encoder[encoded_part])
            else:
                if not part:
                    continue
                # 普通文本区间按照 GPT-2 正则规则预分词
                for token in re.findall(GPT2_SPLIT_PATTERN, part):
                    # 将切分出来的子词字符串用 UTF-8 编码，再映射到安全的 Unicode 字符表示
                    token_bytes = token.encode("utf-8")
                    token_translated = "".join(
                        self.byte_encoder[b] for b in token_bytes
                    )

                    # 对该安全 Unicode 字符流执行 BPE 合并逻辑
                    word_bpe_tokens = self.bpe(token_translated)

                    # 将合并后的每一个 BPE 符号通过 byte_decoder 解码还原回原始字节序列
                    for bpe_token in word_bpe_tokens:
                        original_bytes = bytes(self.byte_decoder[c] for c in bpe_token)
                        # 如果还原后的字节流在词表中，直接输出其 Token ID
                        if original_bytes in self.encoder:
                            bpe_tokens.append(self.encoder[original_bytes])
                        else:
                            # 极端兜底情况：如果由于某种原因未能命中词表，则拆分为原始的单字节 Token 输出
                            for b in original_bytes:
                                bpe_tokens.append(self.encoder[bytes([b])])
        return bpe_tokens
        
    def decode(self, ids: list[int]) -> str:
        """
        将 Token ID 序列解码还原为纯文本字符串。
        
        Args:
            ids: Token ID 列表
        Returns:
            解码后的纯文本字符串，采用 UTF-8 解码，如果存在错误字节则以替换字符（errors="replace"）兜底防错。
        """
        text_bytes = b"".join([self.vocab[idx] for idx in ids])
        return text_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        对传入的文本块迭代器（如逐行读取大文件）进行流式（Generator）BPE 编码。
        避免一次性将全部大文本加载到内存中，提供内存高效的流式处理能力。
        """
        for chunk in iterable:
            yield from self.encode(chunk)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    基于唯一单词词频统计（Word Counts）的快速 BPE 词表训练函数。
    
    Args:
        input_path: 输入的文本训练文件路径
        vocab_size: 目标期望词表大小（必须大于 256 + 外部传入的特殊 Token 数量）
        special_tokens: 声明的特殊 Token 列表（在训练中不会合并）
    Returns:
        vocab: 生成的最终词表字典 dict[int, bytes]
        final_merges: 返回的合并优先级元组列表 list[tuple[bytes, bytes]]
    """
    # 读入全部训练文本
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    # 对训练文本首先进行特殊 Token 的隔离切分，防止训练文本中的特殊 Token 被错误拆分合并
    if special_tokens:
        pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        parts = re.split(pattern, text)
    else:
        parts = [text]
        
    # 按 GPT-2 正则分词规则过滤获取训练用单词序列
    words = []
    for part in parts:
        if part in special_tokens:
            continue
        if not part:
            continue
        words.extend(re.findall(GPT2_SPLIT_PATTERN, part))
        
    # 获取单字节对应的安全 Unicode 映射关系
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    
    # 建立唯一单词的初始词频统计字典 word_counts
    # 键为安全 Unicode 字符组成的元组，值为语料库中对应的频次
    word_counts = {}
    for word in words:
        word_bytes = word.encode("utf-8")
        word_chars = tuple(byte_encoder[b] for b in word_bytes)
        word_counts[word_chars] = word_counts.get(word_chars, 0) + 1
        
    # 基础词表初始化为 256 个单字节：0 -> b'\x00' ... 255 -> b'\xff'
    vocab = {i: bytes([i]) for i in range(256)}
    # 将特殊 Token 依次放入词表中占位
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
        
    merges = []
    
    # 词表训练循环：当词表大小未达到期望的 vocab_size 时循环寻找最佳对并合并
    while len(vocab) < vocab_size:
        pairs = {}
        # 统计所有单词内部相邻 pair 在语料中出现的累积总频次
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] = pairs.get(pair, 0) + count
                
        # 若无可供合并的相邻 pair（说明文本已全部合并为单元素），则提前中断循环
        if not pairs:
            break
            
        # 确定性的冲突解决（Tie-breaking）排序键：
        # 排序标准：1. 累计频数越大越优先；2. 原始字节序列在字典序中越大越优先（以此顺序打破频数冲突平衡）
        def get_stat_key(p):
            b1 = bytes(byte_decoder[c] for c in p[0])
            b2 = bytes(byte_decoder[c] for c in p[1])
            return (pairs[p], b1, b2)
            
        # 选择当前合并价值最高的目标 bigram 字符对
        best = max(pairs, key=get_stat_key)
        merges.append(best)
        
        # 计算新 Token 对应的 Unicode 和原始字节表示，并将其录入词表
        out_token = best[0] + best[1]
        new_token_bytes = bytes(byte_decoder[c] for c in out_token)
        vocab[len(vocab)] = new_token_bytes
        
        # 局部合并更新唯一单词字典：
        # 采用 word.index(best[0], i) 跳转判断，极大提速子词替换循环
        new_word_counts = {}
        for word, count in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # 快速查找到 best[0] 出现的索引位置，跳过不含合并目标的区间
                    j = word.index(best[0], i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                # 判断当前位置是否能与下一位置字符合并为目标 bigram
                if i < len(word) - 1 and word[i] == best[0] and word[i+1] == best[1]:
                    new_word.append(out_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] = new_word_counts.get(tuple(new_word), 0) + count
        word_counts = new_word_counts
        
    # 将训练阶段以 Unicode 字符对形式记录的 merges 转换为原始的字节元组 (tuple[bytes, bytes]) 并返回
    final_merges = []
    for m1, m2 in merges:
        b1 = bytes(byte_decoder[c] for c in m1)
        b2 = bytes(byte_decoder[c] for c in m2)
        final_merges.append((b1, b2))
        
    return vocab, final_merges
