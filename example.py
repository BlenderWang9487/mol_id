import torch

from mol_id import Transformer, TransformerOutput
from mol_id.molformer_tokenizer import MolTranBertTokenizer
from mol_id.utils.helpers import SmilesCollator

device = "cuda"
pretrained = "<pretrained model dir>"

smiles = ["c1cc(oc1)C(=O)N[C@@H]2C[NH+]3CCC2CC3", "c1cc(ccc1CO)C(=O)N2CCCCC2"]
model = Transformer.from_pretrained(pretrained).to(device)
tokenizer = MolTranBertTokenizer()

# because we use flash attention varlen api, so the input shape is different from common transformers input
# we need to use the collator to collate the input to generate the input_ids, cu_seqlens, max_seqlen
collator = SmilesCollator(tokenizer, model.params.max_seq_len)

model_input = collator.collate_impl(smiles)
model_input = model_input.to(device)

with torch.no_grad():
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):  # flash attention only support half precision

        model_output: TransformerOutput = model(
            **(model_input.__dict__), output_logits=False
        )  # logits is for masked language modeling, not needed for other tasks
        print(model_output)

        # you can get cls embedding by this way
        # cls_embs shape: (batch_size, hidden_size)
        cls_embs = model_output.last_hidden_state[model_input.cu_seqlens[:-1], :]
        print(cls_embs)

        # and then you can feed cls_embs to other networks to do downstream tasks
        # logits = downstream_model(cls_embs) ...
