import regex as re
from typing import Iterable
import functools

# Regular expression used for GPT-2 pre-tokenization
GPT2_SPLIT_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

@functools.lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

class BPE:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_token_ids = {
            token: idx
            for idx, token_bytes in vocab.items()
            if (token := token_bytes.decode("utf-8", errors="ignore"))
            in self.special_tokens
        }
        self.encoder = {v: k for k, v in vocab.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        self.bpe_ranks = {}
        for i, (b1, b2) in enumerate(merges):
            s1 = "".join(self.byte_encoder[b] for b in b1)
            s2 = "".join(self.byte_encoder[b] for b in b2)
            self.bpe_ranks[(s1, s2)] = i
        self.cache = {}
        
    def bytes_to_unicode(self):
        return bytes_to_unicode()
        
    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        self.cache[token] = word
        return word
        
    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        if sorted_special_tokens:
            pattern = "(" + "|".join(re.escape(k) for k in sorted_special_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                encoded_part = part.encode("utf-8")
                if encoded_part in self.encoder:
                    bpe_tokens.append(self.encoder[encoded_part])
            else:
                if not part:
                    continue
                # Split by GPT2 pattern
                for token in re.findall(GPT2_SPLIT_PATTERN, part):
                    token_bytes = token.encode("utf-8")
                    token_translated = "".join(
                        self.byte_encoder[b] for b in token_bytes
                    )

                    # Run BPE
                    word_bpe_tokens = self.bpe(token_translated)

                    # Map back to IDs
                    for bpe_token in word_bpe_tokens:
                        original_bytes = bytes(self.byte_decoder[c] for c in bpe_token)
                        if original_bytes in self.encoder:
                            bpe_tokens.append(self.encoder[original_bytes])
                        else:
                            for b in original_bytes:
                                bpe_tokens.append(self.encoder[bytes([b])])
        return bpe_tokens
        
    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join([self.vocab[idx] for idx in ids])
        return text_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
    if special_tokens:
        pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        parts = re.split(pattern, text)
    else:
        parts = [text]
        
    words = []
    for part in parts:
        if part in special_tokens:
            continue
        if not part:
            continue
        words.extend(re.findall(GPT2_SPLIT_PATTERN, part))
        
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    
    word_counts = {}
    for word in words:
        word_bytes = word.encode("utf-8")
        word_chars = tuple(byte_encoder[b] for b in word_bytes)
        word_counts[word_chars] = word_counts.get(word_chars, 0) + 1
        
    vocab = {i: bytes([i]) for i in range(256)}
    for st in special_tokens:
        vocab[len(vocab)] = st.encode("utf-8")
        
    merges = []
    
    while len(vocab) < vocab_size:
        pairs = {}
        for word, count in word_counts.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pairs[pair] = pairs.get(pair, 0) + count
                
        if not pairs:
            break
            
        def get_stat_key(p):
            b1 = bytes(byte_decoder[c] for c in p[0])
            b2 = bytes(byte_decoder[c] for c in p[1])
            return (pairs[p], b1, b2)
            
        best = max(pairs, key=get_stat_key)
        merges.append(best)
        
        out_token = best[0] + best[1]
        new_token_bytes = bytes(byte_decoder[c] for c in out_token)
        vocab[len(vocab)] = new_token_bytes
        
        new_word_counts = {}
        for word, count in word_counts.items():
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # Optimize replacement speed using index
                    j = word.index(best[0], i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                if i < len(word) - 1 and word[i] == best[0] and word[i+1] == best[1]:
                    new_word.append(out_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_counts[tuple(new_word)] = new_word_counts.get(tuple(new_word), 0) + count
        word_counts = new_word_counts
        
    final_merges = []
    for m1, m2 in merges:
        b1 = bytes(byte_decoder[c] for c in m1)
        b2 = bytes(byte_decoder[c] for c in m2)
        final_merges.append((b1, b2))
        
    return vocab, final_merges
