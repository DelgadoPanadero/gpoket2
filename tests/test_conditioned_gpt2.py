import pytest

torch = pytest.importorskip("torch")
from transformers import GPT2Config
from src.application.train.conditioned_gpt2 import ConditionedGPT2

VOCAB = 50
N_EMBD = 16
N_CTX = 32
NUM_POKEMON = 10


@pytest.fixture
def model():
    config = GPT2Config(
        vocab_size=VOCAB,
        n_embd=N_EMBD,
        n_layer=2,
        n_head=2,
        n_ctx=N_CTX,
        n_positions=N_CTX,
    )
    m = ConditionedGPT2(config, num_pokemon=NUM_POKEMON)
    m.eval()
    return m


# --- forward ---


def test_forward_without_pokemon_idx_produces_correct_shape(model):
    ids = torch.randint(0, VOCAB, (1, 8))
    out = model(input_ids=ids)
    assert out.logits.shape == (1, 8, VOCAB)


def test_forward_with_pokemon_idx_produces_correct_shape(model):
    ids = torch.randint(0, VOCAB, (1, 8))
    out = model(input_ids=ids, pokemon_idx=torch.tensor([0]))
    assert out.logits.shape == (1, 8, VOCAB)


def test_different_pokemon_idx_produce_different_logits(model):
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        out0 = model(input_ids=ids, pokemon_idx=torch.tensor([0])).logits
        out1 = model(input_ids=ids, pokemon_idx=torch.tensor([1])).logits
    assert not torch.allclose(out0, out1)


def test_same_pokemon_idx_produces_same_logits(model):
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        out_a = model(input_ids=ids, pokemon_idx=torch.tensor([3])).logits
        out_b = model(input_ids=ids, pokemon_idx=torch.tensor([3])).logits
    assert torch.allclose(out_a, out_b)


def test_forward_without_pokemon_idx_is_deterministic(model):
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        out_a = model(input_ids=ids).logits
        out_b = model(input_ids=ids).logits
    assert torch.allclose(out_a, out_b)


def test_forward_no_nan_without_conditioning(model):
    ids = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        out = model(input_ids=ids).logits
    assert not torch.isnan(out).any()


def test_forward_no_nan_with_conditioning(model):
    ids = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        out = model(input_ids=ids, pokemon_idx=torch.tensor([0])).logits
    assert not torch.isnan(out).any()


def test_forward_batch_with_conditioning(model):
    ids = torch.randint(0, VOCAB, (4, 8))
    idx = torch.tensor([0, 1, 2, 3])
    out = model(input_ids=ids, pokemon_idx=idx)
    assert out.logits.shape == (4, 8, VOCAB)


def test_conditioning_is_additive_not_replacing(model):
    # With conditioning, output must differ from without — it's summed, not replaced
    ids = torch.randint(0, VOCAB, (1, 4))
    with torch.no_grad():
        out_plain = model(input_ids=ids).logits
        out_cond = model(input_ids=ids, pokemon_idx=torch.tensor([0])).logits
    assert not torch.allclose(out_plain, out_cond)


# --- sample_conditioning ---


def test_sample_conditioning_shape(model):
    cond = model.sample_conditioning()
    assert cond.shape == (1, N_EMBD)


def test_sample_conditioning_two_calls_differ(model):
    # Random sampling should produce different results
    results = [model.sample_conditioning() for _ in range(10)]
    all_same = all(torch.allclose(results[0], r) for r in results[1:])
    assert not all_same


# --- interpolate_conditioning ---


def test_interpolate_alpha_zero_equals_embedding_a(model):
    cond = model.interpolate_conditioning(idx_a=0, idx_b=1, alpha=0.0)
    expected = model.conditioning.weight[0].unsqueeze(0)
    assert torch.allclose(cond, expected)


def test_interpolate_alpha_one_equals_embedding_b(model):
    cond = model.interpolate_conditioning(idx_a=0, idx_b=1, alpha=1.0)
    expected = model.conditioning.weight[1].unsqueeze(0)
    assert torch.allclose(cond, expected)


def test_interpolate_alpha_half_is_midpoint(model):
    cond = model.interpolate_conditioning(idx_a=2, idx_b=5, alpha=0.5)
    emb_a = model.conditioning.weight[2]
    emb_b = model.conditioning.weight[5]
    expected = (0.5 * emb_a + 0.5 * emb_b).unsqueeze(0)
    assert torch.allclose(cond, expected)


def test_interpolate_same_index_returns_that_embedding(model):
    cond = model.interpolate_conditioning(idx_a=4, idx_b=4, alpha=0.5)
    expected = model.conditioning.weight[4].unsqueeze(0)
    assert torch.allclose(cond, expected)


def test_interpolate_output_shape(model):
    cond = model.interpolate_conditioning(idx_a=0, idx_b=1, alpha=0.3)
    assert cond.shape == (1, N_EMBD)


# --- prepare_inputs_for_generation ---


def test_prepare_inputs_includes_pokemon_idx_when_provided(model):
    ids = torch.randint(0, VOCAB, (1, 4))
    idx = torch.tensor([7])
    inputs = model.prepare_inputs_for_generation(ids, pokemon_idx=idx)
    assert "pokemon_idx" in inputs
    assert torch.equal(inputs["pokemon_idx"], idx)


def test_prepare_inputs_excludes_pokemon_idx_when_not_provided(model):
    ids = torch.randint(0, VOCAB, (1, 4))
    inputs = model.prepare_inputs_for_generation(ids)
    assert "pokemon_idx" not in inputs


def test_prepare_inputs_preserves_standard_fields(model):
    ids = torch.randint(0, VOCAB, (1, 4))
    inputs = model.prepare_inputs_for_generation(
        ids,
        pokemon_idx=torch.tensor([0]),
    )
    assert "input_ids" in inputs or "inputs_embeds" in inputs
