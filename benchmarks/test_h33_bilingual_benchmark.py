import unittest
import torch
from transformers import BatchEncoding

from h33_bilingual_benchmark import encode_chat


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors):
        return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})


class EncodeChatTest(unittest.TestCase):
    def test_extracts_input_ids_from_batch_encoding(self):
        ids = encode_chat(FakeTokenizer(), "hello")
        self.assertIsInstance(ids, torch.Tensor)
        self.assertEqual(ids.tolist(), [[1, 2, 3]])


if __name__ == "__main__":
    unittest.main()
