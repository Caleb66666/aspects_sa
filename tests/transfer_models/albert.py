# @File: albert
# @Author : Caleb
# @Email: VanderLancer@gmail.com
# @time: 2020/3/16 10:25:41

import unittest
import torch
from transformers import BertTokenizer, AlbertForMaskedLM


class TestAlbertMaskModel(unittest.TestCase):
    def setUp(self):
        super(TestAlbertMaskModel, self).setUp()

        # albert_pre_train = "/Users/Vander/Code/pytorch_col/albert_chinese_base_hf"
        # albert_pre_train = "/Users/Vander/Code/pytorch_col/albert_chinese_large_hf"
        albert_pre_train = "/Users/Vander/Code/pytorch_col/albert_chinese_xlarge_hf"
        self.tokenizer = BertTokenizer.from_pretrained(albert_pre_train)

        self.mask_model = AlbertForMaskedLM.from_pretrained(albert_pre_train)
        self.mask_token = self.tokenizer.mask_token
        self.mask_id = self.tokenizer.mask_token_id

    def test_model_mask_infer(self):
        input_text = f"今天{self.mask_token}{self.mask_token}很好"
        seq_ids = self.tokenizer.encode(input_text, add_special_tokens=True)
        mask_pos_list = [i for i in range(len(seq_ids)) if seq_ids[i] == self.mask_id]

        torch_ids = torch.tensor(seq_ids).unsqueeze(0)
        outputs = self.mask_model(torch_ids, masked_lm_labels=torch_ids)
        loss, prediction_scores = outputs[:2]

        top_k = 1
        batch_index = torch_ids.size(0) - 1
        for mask_pos in mask_pos_list:
            logit_prob = torch.softmax(prediction_scores[0, mask_pos], dim=0).data.tolist()
            prediction_indexes = torch.topk(prediction_scores[batch_index, mask_pos], k=top_k, sorted=True, dim=0)[1]
            for idx in prediction_indexes:
                prediction_token = self.tokenizer.convert_ids_to_tokens([idx])[0]
                print(idx.item(), prediction_token, logit_prob[idx.item()])
