import sys
import os

# Add the parent directory (assignment1-basics) to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now use absolute import
from cs336_basics.bpe_tokenizer import bpe_tokenizer


cases = ["abc", "aa<|endoftext|>a", "", "你好好", "aaa", "abab"]
for case in cases:
    with open("tmp.txt", "w") as f:
        f.write(case)
    print("=========" + case)
    tokenizer = bpe_tokenizer()
    tokenizer.train_bpe("tmp.txt", vocab_size=260, special_tokens=["<|endoftext|>"])
    print("merges:", tokenizer.merges)
    print("vocab:", tokenizer.vocab)

'''
  text = "abc"
  initial = [b"a", b"b", b"c"]
  merges = [(b"a", b"b"), (b"ab", b"c")]

  你期望最终 token list 是：

  [b"abc"]
'''
token_list = tokenizer.encode("abc")
print(token_list)