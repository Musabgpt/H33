import unittest

from training.train_partial_sft import select_trainable_parameters


class FakeParam:
    def __init__(self, name, size=10):
        self.name = name
        self.requires_grad = True
        self._size = size

    def numel(self):
        return self._size


class FakeModule:
    def __init__(self, *params):
        self._params = list(params)

    def parameters(self):
        return iter(self._params)


class FakeBackbone:
    def __init__(self, layer_count=8):
        self.layers = [
            FakeModule(FakeParam(f"model.layers.{i}.weight", 100))
            for i in range(layer_count)
        ]
        self.norm = FakeModule(FakeParam("model.norm.weight", 10))
        self.embed_tokens = FakeModule(FakeParam("model.embed_tokens.weight", 500))


class FakeConfig:
    tie_word_embeddings = True


class FakeModel:
    def __init__(self):
        self.model = FakeBackbone()
        # Simulate tied lm_head: same Parameter object as embed_tokens.
        self.lm_head = FakeModule(self.model.embed_tokens._params[0])
        self.config = FakeConfig()

    def parameters(self):
        seen = set()
        for layer in self.model.layers:
            for p in layer.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p
        for module in (self.model.norm, self.model.embed_tokens, self.lm_head):
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def named_parameters(self):
        for p in self.parameters():
            yield p.name, p


class TrainableSelectionTest(unittest.TestCase):
    def test_opens_only_last_two_original_blocks_and_final_norm(self):
        model = FakeModel()

        report = select_trainable_parameters(model, last_n_blocks=2)

        trainable = {
            name
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.assertEqual(
            {
                "model.layers.6.weight",
                "model.layers.7.weight",
                "model.norm.weight",
            },
            trainable,
        )
        self.assertFalse(model.model.embed_tokens._params[0].requires_grad)
        self.assertFalse(next(model.lm_head.parameters()).requires_grad)
        self.assertLess(report["ratio"], 0.20)
        self.assertEqual(3, report["trainable_tensors"])

    def test_refuses_zero_or_too_many_blocks(self):
        model = FakeModel()
        with self.assertRaises(ValueError):
            select_trainable_parameters(model, last_n_blocks=0)
        with self.assertRaises(ValueError):
            select_trainable_parameters(model, last_n_blocks=99)

    def test_hard_guard_rejects_unexpected_large_trainable_ratio(self):
        # With just two layers, opening both would exceed the safety ceiling.
        model = FakeModel()
        model.model.layers = model.model.layers[-2:]
        with self.assertRaisesRegex(ValueError, "unsafe trainable ratio"):
            select_trainable_parameters(model, last_n_blocks=2, max_ratio=0.20)


if __name__ == "__main__":
    unittest.main()
