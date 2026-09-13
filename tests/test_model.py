import torch

from brits_mi.model import BRITSMI, torch_marker_gaps


def test_downstream_loss_reaches_recurrent_imputer():
    torch.manual_seed(3)
    model = BRITSMI(n_markers=3, n_static=3, n_context=2, hidden_size=8)
    values = torch.randn(12, 6, 3)
    mask = (torch.rand(12, 6, 3) > 0.35).float()
    times = torch.tensor([0.0, 0.12, 0.29, 0.51, 0.76, 1.0]).repeat(12, 1)
    static = torch.randn(12, 3)
    context = torch.randn(12, 2)
    output = model(
        values * mask,
        mask,
        torch_marker_gaps(mask, times),
        times,
        static,
        context,
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output["logits"], torch.randint(0, 2, (12,)).float()
    )
    loss.backward()
    gradient = model.forward_rits.history_head.weight.grad
    assert gradient is not None
    assert float(gradient.abs().sum()) > 0.0

