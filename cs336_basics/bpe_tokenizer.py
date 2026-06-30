import regex

class bpe_tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

    def decode(self, tokens):
        # list to string
        data_bytes = b"".join(self.vocab[token] for token in tokens)
        return data_bytes.decode("utf-8")

    def encode(self, text):
        #string to list
        #special token split
        special_pattern = "(" + "|".join(regex.escape(tok) for tok in self.special_tokens) + ")"
        parts = regex.split(special_pattern, text)
        bytes_to_id = {v: k for k, v in self.vocab.items()}
        tokens = []
        for part in parts:    
            if part == "":
                continue
            if part in self.special_tokens:
                text_bytes = part.encode("utf-8")
                tokens.append(bytes_to_id[text_bytes])
            else:
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                for pretoken in regex.findall(PAT, part):
                    #string to list of byte
                    text_bytes = pretoken.encode("utf-8")
                    current_token_list = [bytes([b]) for b in text_bytes]
                    for (pair0, pair1) in self.merges:
                        i = 0
                        while i < len(current_token_list) - 1:
                            if current_token_list[i] == pair0 and current_token_list[i+1] == pair1:
                                current_token_list[i:i+2] = [current_token_list[i] + current_token_list[i+1]]
                            else:
                                i += 1
                    #整个pretoken 匹配完，转id
                    for subtoken in current_token_list:
                        tokens.append(bytes_to_id[subtoken])
        return tokens

    def train_bpe(self, input_path, vocab_size, special_tokens):
        self.special_tokens = special_tokens if special_tokens is not None else []
        special_token_sorted = sorted(self.special_tokens, key = lambda x: len(x), reverse = True)
        self.vocab = {}
        self.merges = []

        # 1. Read input as bytes
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        for i in range(256):
            self.vocab[i] = bytes([i])
        for i in range(len(special_token_sorted)):
            self.vocab[256 + i] = special_token_sorted[i].encode("utf-8")
        
        # 2. pretokenize
        pretokens = []  # ['hello', 'world']
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if len(special_token_sorted) > 0:
            special_token_pattern = "(" + "|".join(regex.escape(tok) for tok in special_token_sorted) + ")"
            parts = regex.split(special_token_pattern, text)
            #print("parts:", parts)
        else:
            parts = [text]

        for part in parts:
            #print("part:", part)
            if part in self.special_tokens or part == "":
                continue
            for m in regex.finditer(PAT, part):
                pretoken = m.group(0)
                pretokens.append(pretoken) 
        pretokens_frequency = {} # {'hello':1, 'world':1}
        for pretoken in pretokens:
            if pretoken in pretokens_frequency:
                pretokens_frequency[pretoken] += 1
            else:
                pretokens_frequency[pretoken] = 1
        #print("pretokens:", pretokens)
        #print("pretokens_frequency:", pretokens_frequency)
        
        # 3. train loop
        pretoken_splits = {} # {'hello': [b"h", b"e", b"l", b"l", b"o"]}
        for pretoken in pretokens_frequency.keys():
            pretoken_bytes = pretoken.encode("utf-8")
            byte_list = [pretoken_bytes[i:i+1] for i in range(len(pretoken_bytes))] #[b"h", b"e", b"l", b"l", b"o"]
            pretoken_splits[pretoken] = byte_list

        #print("pretoken_splits:", pretoken_splits)
        
        next_token_id = len(self.vocab)
        while len(self.vocab) < vocab_size:
            # 3.1 update pretoken splits
            pair_counts = {} # {(b"h", b"e"): 1, (b"e", b"l"): 1}
            #iterate through pretokens
            for pretoken, frequency in pretokens_frequency.items():
                #iterate through pretoken splits
                #print("pretoken:", pretoken)
                for i in range(1, len(pretoken_splits[pretoken])):
                    pair = (pretoken_splits[pretoken][i-1], pretoken_splits[pretoken][i]) # (b"h", b"e")
                    #print("pair:", pair)
                    if pair in pair_counts:
                        pair_counts[pair] += frequency
                    else:
                        pair_counts[pair] = frequency
            if len(pair_counts) == 0:
                break # no more pairs to merge
            
            #lexical order if tie
            most_common_pair, frequency = max(pair_counts.items(), key=lambda x:(x[1], x[0]))
            # 加入vocab
            self.vocab[next_token_id] = most_common_pair[0] + most_common_pair[1]
            next_token_id += 1
            # 加入merges
            self.merges.append(most_common_pair)

            # 3.2.1 update merge to pretoken splits
            
            for pretoken, old_split in pretoken_splits.items():
                i = 0
                new_splits = []
                while i < len(old_split):
                    if i < len(old_split) - 1 and (old_split[i], old_split[i+1]) == most_common_pair:
                        new_splits.append(most_common_pair[0] + most_common_pair[1])
                        i += 2 # skip tokens to be merged
                    else:
                        new_splits.append(old_split[i])
                        i += 1
                pretoken_splits[pretoken] = new_splits
        
        return self.vocab, self.merges

    def encode_iterable(self, iterable):
        for text_chunk in iterable:
            for token_id in self.encode(text_chunk):
                yield token_id
            