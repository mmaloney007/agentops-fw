import torch

from agent_stable_slo.utils.dist import rank_world, seed_with_rank


def _train_single_batch(model, opt, x, y):
    opt.zero_grad()
    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    opt.step()
    return model.weight.detach().clone(), loss.item()


def _train_grad_accum(model, opt, x, y, grad_accum=2):
    opt.zero_grad()
    total_loss = 0.0
    for i in range(x.shape[0]):
        pred = model(x[i : i + 1])
        loss = torch.nn.functional.mse_loss(pred, y[i : i + 1]) / grad_accum
        loss.backward()
        total_loss += loss.item()
        if (i + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad()
    return model.weight.detach().clone(), total_loss


def test_grad_accum_matches_full_batch():
    torch.manual_seed(42)
    x = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
    y = torch.tensor([[0.3], [-0.7]])

    model_full = torch.nn.Linear(2, 1, bias=False)
    model_accum = torch.nn.Linear(2, 1, bias=False)
    model_accum.load_state_dict(model_full.state_dict())

    opt_full = torch.optim.SGD(model_full.parameters(), lr=0.1)
    opt_accum = torch.optim.SGD(model_accum.parameters(), lr=0.1)

    w_full, loss_full = _train_single_batch(model_full, opt_full, x, y)
    w_accum, loss_accum = _train_grad_accum(model_accum, opt_accum, x, y, grad_accum=2)

    assert torch.allclose(w_full, w_accum, atol=1e-6)
    assert abs(loss_full - loss_accum) < 1e-6


def test_rank_seed_adjustment():
    # default env should yield rank 0, world 1
    r, w = rank_world()
    assert r == 0
    assert w == 1
    assert seed_with_rank(10) == 10
