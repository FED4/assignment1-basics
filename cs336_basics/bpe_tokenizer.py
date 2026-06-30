from collections import Counter, defaultdict

import regex


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class bpe_tokenizer:
    def __init__(self, vocab=None, merges=None, special_tokens=None):
        self.vocab = vocab or {}
        self.merges = merges or []
        self.special_tokens = special_tokens if special_tokens is not None else []
        self._refresh_indexes()

    def _refresh_indexes(self):
        self.bytes_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_token_bytes = {token: token.encode("utf-8") for token in self.special_tokens}
        self.special_token_pattern = self._make_special_token_pattern()

    def _make_special_token_pattern(self):
        if not self.special_tokens:
            return None
        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        return "(" + "|".join(regex.escape(token) for token in special_tokens) + ")"

    def decode(self, tokens):
        data_bytes = b"".join(self.vocab[token] for token in tokens)
        return data_bytes.decode("utf-8", errors="replace")

    def _encode_pretoken(self, pretoken):
        parts = [bytes([byte]) for byte in pretoken.encode("utf-8")]
        while len(parts) >= 2:
            best_rank = None
            best_index = None
            for i in range(len(parts) - 1):
                rank = self.merge_ranks.get((parts[i], parts[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_index is None:
                break
            parts[best_index : best_index + 2] = [parts[best_index] + parts[best_index + 1]]
        return [self.bytes_to_id[part] for part in parts]

    def encode(self, text):
        tokens = []
        parts = regex.split(self.special_token_pattern, text) if self.special_token_pattern else [text]
        for part in parts:
            if part == "":
                continue
            if part in self.special_tokens:
                tokens.append(self.bytes_to_id[self.special_token_bytes[part]])
                continue
            for pretoken in regex.findall(PAT, part):
                tokens.extend(self._encode_pretoken(pretoken))
        return tokens

    def encode_iterable(self, iterable):
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def train_bpe(self, input_path, vocab_size, special_tokens):
        self.special_tokens = special_tokens if special_tokens is not None else []
        special_token_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = []

        for i, special_token in enumerate(special_token_sorted):
            self.vocab[256 + i] = special_token.encode("utf-8")

        with open(input_path, encoding="utf-8") as f:
            text = f.read()

        if special_token_sorted:
            special_token_pattern = "(" + "|".join(regex.escape(token) for token in special_token_sorted) + ")"
            text_parts = regex.split(special_token_pattern, text)
        else:
            text_parts = [text]

        pretoken_counts = Counter()
        for part in text_parts:
            if part == "" or part in self.special_tokens:
                continue
            pretoken_counts.update(regex.findall(PAT, part))

        word_splits = []
        word_frequencies = []
        for pretoken, frequency in pretoken_counts.items():
            pretoken_bytes = pretoken.encode("utf-8")
            word_splits.append([pretoken_bytes[i : i + 1] for i in range(len(pretoken_bytes))])
            word_frequencies.append(frequency)

        pair_counts = Counter()
        pair_to_words = defaultdict(set)
        for word_index, split in enumerate(word_splits):
            frequency = word_frequencies[word_index]
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_counts[pair] += frequency
                pair_to_words[pair].add(word_index)

        next_token_id = len(self.vocab)
        while len(self.vocab) < vocab_size:
            if not pair_counts:
                break

            most_common_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
            self.vocab[next_token_id] = most_common_pair[0] + most_common_pair[1]
            next_token_id += 1
            self.merges.append(most_common_pair)

            affected_word_indices = list(pair_to_words[most_common_pair])
            for word_index in affected_word_indices:
                old_split = word_splits[word_index]
                frequency = word_frequencies[word_index]
                old_pairs = set()
                for i in range(len(old_split) - 1):
                    pair = (old_split[i], old_split[i + 1])
                    old_pairs.add(pair)
                    pair_counts[pair] -= frequency
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]
                for pair in old_pairs:
                    pair_to_words[pair].discard(word_index)

                i = 0
                new_split = []
                while i < len(old_split):
                    if i < len(old_split) - 1 and (old_split[i], old_split[i + 1]) == most_common_pair:
                        new_split.append(most_common_pair[0] + most_common_pair[1])
                        i += 2
                    else:
                        new_split.append(old_split[i])
                        i += 1
                word_splits[word_index] = new_split

                for i in range(len(new_split) - 1):
                    pair = (new_split[i], new_split[i + 1])
                    pair_counts[pair] += frequency
                    pair_to_words[pair].add(word_index)

        self._refresh_indexes()
        return self.vocab, self.merges
