import pytest
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")
from src.application.train.pokemon_trainer import ForCausalLMLossWeighed

VOCAB = 10


def mock_output(logits_tensor):
    out = MagicMock()
    out.logits = logits_tensor
    return out


def uniform_logits(batch=1, seq=4):
    return mock_output(torch.zeros(batch, seq, VOCAB))


def random_logits(batch=1, seq=4):
    return mock_output(torch.randn(batch, seq, VOCAB))


# --- output shape ---


def test_loss_is_scalar():
    loss = ForCausalLMLossWeighed(
        uniform_logits(),
        torch.zeros(1, 4, dtype=torch.long),
        vocab_size=VOCAB,
    )
    assert loss.dim() == 0


def test_loss_is_non_negative():
    loss = ForCausalLMLossWeighed(
        random_logits(),
        torch.randint(0, VOCAB, (1, 4)),
        vocab_size=VOCAB,
    )
    assert loss.item() >= 0


# --- ignore index ---


def test_all_ignore_index_produces_zero_or_nan():
    labels = torch.full((1, 4), -100, dtype=torch.long)
    loss = ForCausalLMLossWeighed(uniform_logits(), labels, vocab_size=VOCAB)
    assert loss.item() == 0.0 or torch.isnan(loss)


def test_partial_ignore_index_does_not_nan():
    labels = torch.tensor([[0, -100, 2, -100]])
    loss = ForCausalLMLossWeighed(random_logits(), labels, vocab_size=VOCAB)
    assert not torch.isnan(loss)


# --- token weighting ---


def test_token_weight_one_same_as_no_weighting():
    logits_t = torch.randn(2, 4, VOCAB)
    labels = torch.randint(0, VOCAB, (2, 4))

    loss_uniform = ForCausalLMLossWeighed(
        mock_output(logits_t),
        labels,
        vocab_size=VOCAB,
    )
    loss_weight1 = ForCausalLMLossWeighed(
        mock_output(logits_t),
        labels,
        vocab_size=VOCAB,
        weight_token_id=3,
        token_weight=1.0,
    )
    assert torch.allclose(loss_uniform, loss_weight1, atol=1e-5)


def test_token_weight_zero_reduces_loss_when_target_is_that_token():
    # Model is completely wrong: predicts token 0, target is token 1
    logits_t = torch.full((1, 4, VOCAB), -1e9)
    logits_t[:, :, 0] = 1e9
    labels = torch.ones(1, 4, dtype=torch.long)  # target = token 1

    loss_normal = ForCausalLMLossWeighed(
        mock_output(logits_t),
        labels,
        vocab_size=VOCAB,
    )
    loss_zero_weight = ForCausalLMLossWeighed(
        mock_output(logits_t),
        labels,
        vocab_size=VOCAB,
        weight_token_id=1,
        token_weight=0.0,
    )
    assert loss_zero_weight.item() < loss_normal.item()


def test_no_weight_token_id_does_not_crash():
    loss = ForCausalLMLossWeighed(
        random_logits(),
        torch.randint(0, VOCAB, (1, 4)),
        vocab_size=VOCAB,
        weight_token_id=None,
    )
    assert not torch.isnan(loss)


def test_higher_token_weight_increases_loss_for_that_token():
    # Model predicts token 0 for everything, target = token 2
    logits_t = torch.full((1, 8, VOCAB), -1e9)
    logits_t[:, :, 0] = 1e9
    labels = torch.full((1, 8), 2, dtype=torch.long)

    loss_low = ForCausalLMLossWeighed(
        mock_output(logits_t.clone()),
        labels,
        vocab_size=VOCAB,
        weight_token_id=2,
        token_weight=0.1,
    )
    loss_high = ForCausalLMLossWeighed(
        mock_output(logits_t.clone()),
        labels,
        vocab_size=VOCAB,
        weight_token_id=2,
        token_weight=1.0,
    )
    assert loss_high.item() > loss_low.item()


# --- batch reduction ---


def test_loss_with_num_items_in_batch():
    logits_t = torch.randn(2, 4, VOCAB)
    labels = torch.randint(0, VOCAB, (2, 4))
    num_items = torch.tensor(8)

    loss = ForCausalLMLossWeighed(
        mock_output(logits_t),
        labels,
        vocab_size=VOCAB,
        num_items_in_batch=num_items,
    )
    assert not torch.isnan(loss)
    assert loss.item() >= 0
