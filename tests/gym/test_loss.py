import pytest
from unittest.mock import MagicMock

torch = pytest.importorskip("torch")
from src.application.gym.model.for_causal_lm_loss_weighed import ForCausalLMLossWeighed

VOCAB = 10


def mock_output(logits):
    out = MagicMock()
    out.logits = logits
    return out


class TestLoss:
    def test_scalar_and_non_negative(self):
        loss = ForCausalLMLossWeighed(
            mock_output(torch.randn(1, 4, VOCAB)),
            torch.randint(0, VOCAB, (1, 4)),
            vocab_size=VOCAB,
        )
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_all_ignore_index_produces_zero(self):
        loss = ForCausalLMLossWeighed(
            mock_output(torch.zeros(1, 4, VOCAB)),
            torch.full((1, 4), -100, dtype=torch.long),
            vocab_size=VOCAB,
        )
        assert loss.item() == 0.0 or torch.isnan(loss)


class TestTokenWeighting:
    def test_higher_weight_increases_loss_for_that_token(self):
        logits = torch.full((1, 8, VOCAB), -1e9)
        logits[:, :, 0] = 1e9  # model always predicts token 0
        labels = torch.full((1, 8), 2, dtype=torch.long)  # target is token 2

        loss_low = ForCausalLMLossWeighed(
            mock_output(logits.clone()), labels, vocab_size=VOCAB,
            weight_token_id=2, token_weight=0.1,
        )
        loss_high = ForCausalLMLossWeighed(
            mock_output(logits.clone()), labels, vocab_size=VOCAB,
            weight_token_id=2, token_weight=1.0,
        )
        assert loss_high.item() > loss_low.item()
