"""Quick smoke test for LocalizedRFProposal on GPU."""
import torch
from proposals.localized_rf import LocalizedRFProposal

torch.manual_seed(0)

for arch in ["local_mlp", "local_resnet1d"]:
    print(f"--- {arch} ---")
    model = LocalizedRFProposal(
        radius=3,
        architecture=arch,
        state_dim=20,
        use_observations=True,
        obs_components=list(range(20)),
        predict_delta=False,
        num_sampling_steps=5,
        num_likelihood_steps=5,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print("params:", sum(p.numel() for p in model.parameters()))

    device = next(model.parameters()).device
    x_prev = torch.randn(4, 20, device=device)
    y = torch.randn(4, 20, device=device)

    x_sampled = model.sample(x_prev, y)
    print("sample shape:", x_sampled.shape)

    x2, ell = model.sample_and_per_dim_log_prob(x_prev, y)
    print("x2 shape:", x2.shape, "ell shape:", ell.shape)
    print("sum ell  :", ell.sum(-1).tolist())

    lp = model.log_prob(x2, x_prev, y)
    print("backwd lp:", lp.tolist())
    print("diff     :", (ell.sum(-1) - lp).abs().max().item())

    # Predict-delta variant
    print("-- predict_delta --")
    model2 = LocalizedRFProposal(
        radius=3,
        architecture=arch,
        state_dim=20,
        use_observations=True,
        obs_components=list(range(20)),
        predict_delta=True,
        num_sampling_steps=5,
        num_likelihood_steps=5,
    )
    if torch.cuda.is_available():
        model2 = model2.cuda()
    x3, ell2 = model2.sample_and_per_dim_log_prob(x_prev, y)
    print("x3 shape:", x3.shape, "ell2 shape:", ell2.shape)
    lp2 = model2.log_prob(x3, x_prev, y)
    print("sum ell2:", ell2.sum(-1).tolist())
    print("backwd  :", lp2.tolist())

print("All smoke tests passed.")
