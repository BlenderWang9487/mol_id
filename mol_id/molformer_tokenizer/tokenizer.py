from pathlib import Path

import regex as re
from transformers import BertTokenizer

PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"


class MolTranBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file: str = str(Path(__file__).parent / "bert_vocab.txt"),
        do_lower_case=False,
        unk_token="<unk>",
        sep_token="<eos>",
        pad_token="<pad>",
        cls_token="<bos>",
        mask_token="<mask>",
        **kwargs
    ):
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )

        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string


if __name__ == "__main__":
    tokenizer = MolTranBertTokenizer()
    batch = tokenizer(
        ["CC(C)C(=O)O", "CC"],
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
        return_attention_mask=True,
    )
    print(batch)
