import pytest

torch = pytest.importorskip("torch")
from transformers import GPT2Config
from src.application.gym.model.conditioned_gpt2 import ConditionedGPT2

VOCAB, N_EMBD, N_CTX, NUM_POKEMON = 50, 16, 32, 10


@pytest.fixture
def model():
    config = GPT2Config(
        vocab_size=VOCAB, n_embd=N_EMBD, n_layer=2, n_head=2,
        n_ctx=N_CTX, n_positions=N_CTX,
    )
    m = ConditionedGPT2(config, num_pokemon=NUM_POKEMON)
    m.eval()
    return m


class TestForward:
    def test_output_shape_with_and_without_conditioning(self, model):
        ids = torch.randint(0, VOCAB, (1, 8))
        assert model(input_ids=ids).logits.shape == (1, 8, VOCAB)
        assert model(input_ids=ids, pokemon_idx=torch.tensor([0])).logits.shape == (1, 8, VOCAB)

    def test_different_pokemon_idx_produce_different_logits(self, model):
        ids = torch.randint(0, VOCAB, (1, 8))
        with torch.no_grad():
            out0 = model(input_ids=ids, pokemon_idx=torch.tensor([0])).logits
            out1 = model(input_ids=ids, pokemon_idx=torch.tensor([1])).logits
        assert not torch.allclose(out0, out1)


class TestSampleConditioning:
    def test_shape_and_varies_across_calls(self, model):
        results = [model.sample_conditioning() for _ in range(5)]
        assert results[0].shape == (1, N_EMBD)
        assert not all(torch.allclose(results[0], r) for r in results[1:])


class TestInterpolateConditioning:
    def test_alpha_zero_returns_embedding_a(self, model):
        cond = model.interpolate_conditioning(idx_a=0, idx_b=1, alpha=0.0)
        assert torch.allclose(cond, model.conditioning.weight[0].unsqueeze(0))

    def test_alpha_one_returns_embedding_b(self, model):
        cond = model.interpolate_conditioning(idx_a=0, idx_b=1, alpha=1.0)
        assert torch.allclose(cond, model.conditioning.weight[1].unsqueeze(0))


class TestPrepareInputsForGeneration:
    def test_pokemon_idx_included_when_provided_excluded_otherwise(self, model):
        ids = torch.randint(0, VOCAB, (1, 4))
        assert "pokemon_idx" in model.prepare_inputs_for_generation(ids, pokemon_idx=torch.tensor([3]))
        assert "pokemon_idx" not in model.prepare_inputs_for_generation(ids)
