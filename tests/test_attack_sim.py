"""
Tests for src/attack_sim.py.

Run with ``pytest tests`` or ``python tests/test_attack_sim.py`` from the repository root.
"""

import os
import sys
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attack_sim import (LossThresholdAttack, auroc, best_threshold, load_model, roc_curve,  # noqa: E402
                            save_results, tpr_at_fpr)


def make_data(n=200, d=20, classes=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g), torch.randint(0, classes, (n,), generator=g)


def make_mlp(d=20, classes=3, dropout=0.0):
    return nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, classes))


def overfit(model, x, y, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model


# -- metrics -------------------------------------------------------------------------------

def test_auroc_extremes_and_ties():
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auroc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auroc([1, 1, 1], [1, 1, 1]) == 0.5
    # of the 9 pairs, 6 are won and 2 are tied (half each): 7 / 9
    assert abs(auroc([1, 2, 3], [0, 1, 2]) - 7.0 / 9.0) < 1e-12


def test_auroc_matches_sklearn_if_available():
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(0.3, 1, 300).round(1), rng.normal(0, 1, 500).round(1)  # rounding creates ties
    expected = roc_auc_score(np.r_[np.ones(300), np.zeros(500)], np.r_[pos, neg])
    assert abs(auroc(pos, neg) - expected) < 1e-12
    fpr, tpr = roc_curve(pos, neg)
    assert abs(np.trapz(tpr, fpr) - expected) < 1e-12


def test_tpr_at_fpr_and_threshold():
    pos, neg = np.arange(10, 20), np.arange(0, 10)
    assert tpr_at_fpr(pos, neg, 0.0) == 1.0
    losses_m, losses_n = np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])
    thr, bal = best_threshold(losses_m, losses_n)
    assert bal == 1.0 and 0.3 <= thr < 0.4


# -- the attack on models ------------------------------------------------------------------

def test_overfit_model_leaks_and_clean_model_does_not():
    x, y = make_data(400)
    (xm, ym), (xn, yn) = (x[:200], y[:200]), (x[200:], y[200:])
    attack = LossThresholdAttack()

    model = overfit(make_mlp(), xm, ym)
    leaky = attack.run((xm, ym), (xn, yn), model=model)
    assert leaky["auroc"] > 0.9 and leaky["loss_gap"] > 0

    fresh = attack.run((xm, ym), (xn, yn), model=make_mlp())  # untrained: no membership signal
    assert abs(fresh["auroc"] - 0.5) < 0.1


def test_all_input_formats_agree():
    x, y = make_data(120)
    xn, yn = make_data(80, seed=1)
    model = make_mlp(dropout=0.5)
    attack = LossThresholdAttack(batch_size=32)
    reference = attack.losses((x, y), model)

    with torch.no_grad():
        logits = model.eval()(x)
    formats = {
        "dataloader": DataLoader(TensorDataset(x, y), batch_size=7),
        "dataset": TensorDataset(x, y),
        "list of samples": [(x[i], y[i]) for i in range(len(x))],
        "numpy tuple": (x.numpy(), y.numpy()),
        "dict batches": ({"image": x[i:i + 10], "label": y[i:i + 10]} for i in range(0, len(x), 10)),
        "triple batches": ((x[i:i + 10], y[i:i + 10], torch.arange(10)) for i in range(0, len(x), 10)),
        "logits": {"logits": logits, "labels": y},
        "losses": reference,
        "losses dict": {"losses": torch.as_tensor(reference)},
    }
    for name, data in formats.items():
        got = attack.losses(data, model)
        assert np.allclose(got, reference, atol=1e-5), name

    # a plain callable returning numpy, and a tuple output, are fine too
    def numpy_model(inputs):
        return (model.eval()(inputs).detach().numpy(), "extra")
    assert np.allclose(attack.losses((x, y), numpy_model), reference, atol=1e-5)
    assert attack.run((x, y), (xn, yn), model=numpy_model)["n_nonmembers"] == 80


def test_eval_mode_is_used_and_restored():
    x, y = make_data(60)
    model = make_mlp(dropout=0.9).train()
    attack = LossThresholdAttack()
    a, b = attack.losses((x, y), model), attack.losses((x, y), model)
    assert np.array_equal(a, b)  # dropout is off, so repeated runs match
    assert model.training  # and the caller's mode is put back


def test_multilabel_with_ignored_labels():
    logits = torch.randn(50, 14)
    labels = torch.randint(0, 2, (50, 14)).float()
    labels[:, 3] = -1  # "uncertain" label for everyone
    attack = LossThresholdAttack(ignore_label=-1)
    got = attack.losses({"logits": logits, "labels": labels})

    keep = [i for i in range(14) if i != 3]
    expected = nn.functional.binary_cross_entropy_with_logits(
        logits[:, keep], labels[:, keep], reduction="none").mean(dim=1)
    assert np.allclose(got, expected.numpy(), atol=1e-5)

    total = LossThresholdAttack(ignore_label=-1, label_reduction="sum").losses({"logits": logits, "labels": labels})
    assert np.allclose(total, got * 13, atol=1e-4)


def test_probabilities_binary_and_custom_loss():
    logits = torch.randn(40, 4)
    y = torch.randint(0, 4, (40,))
    from_logits = LossThresholdAttack().losses({"logits": logits, "labels": y})
    from_probs = LossThresholdAttack(from_probs=True).losses({"logits": logits.softmax(1), "labels": y})
    assert np.allclose(from_logits, from_probs, atol=1e-4)

    onehot = nn.functional.one_hot(y, 4).float()
    assert np.allclose(LossThresholdAttack(loss_fn="ce").losses({"logits": logits, "labels": onehot}),
                       from_logits, atol=1e-5)

    single = torch.randn(30, 1)
    labels = torch.randint(0, 2, (30,))
    binary = LossThresholdAttack().losses({"logits": single, "labels": labels})
    assert binary.shape == (30,)

    custom = LossThresholdAttack(loss_fn=lambda lg, lb: (lg.argmax(1) != lb).float())
    assert set(np.unique(custom.losses({"logits": logits, "labels": y}))) <= {0.0, 1.0}


def test_threshold_sources():
    x, y = make_data(300)
    model = overfit(make_mlp(), x[:100], y[:100])
    members, nonmembers = (x[:100], y[:100]), (x[100:200], y[100:200])
    calibration = ((x[:100], y[:100]), (x[200:], y[200:]))
    attack = LossThresholdAttack()

    oracle = attack.run(members, nonmembers, model=model)
    assert oracle["threshold_source"] == "oracle"
    assert oracle["accuracy"] == oracle["accuracy_oracle"]

    calibrated = attack.run(members, nonmembers, model=model, calibration=calibration)
    assert calibrated["threshold_source"] == "calibration"
    assert calibrated["accuracy"] <= calibrated["accuracy_oracle"] + 1e-12

    given = attack.run(members, nonmembers, model=model, threshold=1e9)  # everything is a "member"
    assert given["threshold_source"] == "given" and abs(given["balanced_accuracy"] - 0.5) < 1e-12


def test_bootstrap_with_groups():
    rng = np.random.default_rng(0)
    members = {"losses": rng.normal(0.5, 0.3, 300).clip(0)}
    nonmembers = {"losses": rng.normal(0.8, 0.3, 300).clip(0)}
    patients_m, patients_n = np.repeat(np.arange(100), 3), np.repeat(np.arange(100, 200), 3)

    attack = LossThresholdAttack(n_bootstrap=200, seed=1)
    plain = attack.run(members, nonmembers)
    grouped = attack.run(members, nonmembers, member_groups=patients_m, nonmember_groups=patients_n)
    for res in (plain, grouped):
        low, high = res["auroc_ci"]
        assert low < res["auroc"] < high
    # resampling whole patients gives a wider interval than treating samples as independent
    assert grouped["auroc_ci"][1] - grouped["auroc_ci"][0] > plain["auroc_ci"][1] - plain["auroc_ci"][0] - 0.01

    try:
        attack.run(members, nonmembers, member_groups=patients_m[:-1])
        raise AssertionError("expected a length error")
    except ValueError:
        pass

    x, y = make_data(30)
    shuffled = DataLoader(TensorDataset(x, y), batch_size=5, shuffle=True)
    try:
        attack.run(shuffled, (x, y), model=make_mlp(), member_groups=np.arange(30))
        raise AssertionError("expected a shuffle error")
    except ValueError:
        pass


def test_checkpoints_per_client_and_saving():
    x, y = make_data(200)
    model = overfit(make_mlp(), x[:100], y[:100], epochs=100)
    attack = LossThresholdAttack()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        torch.save(model.state_dict(), path)
        wrapped = os.path.join(tmp, "ckpt.pt")
        torch.save({"state_dict": model.state_dict(), "round": 3}, wrapped)

        loaded = load_model(path, make_mlp)
        assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))

        results = attack.run_checkpoints(
            {"from_path": path, "from_wrapped": wrapped, "from_dict": model.state_dict(), "from_module": model},
            (x[:100], y[:100]), (x[100:], y[100:]), model_factory=make_mlp)
        assert len({round(r["auroc"], 9) for r in results.values()}) == 1

        clients = {"hospital_0": (x[:50], y[:50]), "hospital_1": (x[50:100], y[50:100])}
        per_client = attack.run_per_client(clients, (x[100:], y[100:]), model=path, model_factory=make_mlp)
        assert set(per_client) == {"hospital_0", "hospital_1"}
        assert all(r["n_nonmembers"] == 100 for r in per_client.values())

        out = os.path.join(tmp, "nested", "res.json")
        save_results({"fedavg": attack.run((x[:100], y[:100]), (x[100:], y[100:]), model=model, return_losses=True)}, out)
        assert os.path.getsize(out) > 0


def test_bad_inputs_raise_clear_errors():
    x, y = make_data(20)
    attack = LossThresholdAttack()
    for bad, exc in [((x, y), ValueError),                        # no model
                     ({"losses": [0.1, float("nan")]}, ValueError),
                     ({"foo": 1}, KeyError)]:
        try:
            attack.losses(bad)
            raise AssertionError("expected an error")
        except exc:
            pass
    try:
        load_model({"a": torch.zeros(1)}, None)
        raise AssertionError("expected an error")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = {k: v for k, v in sorted(globals().items()) if k.startswith("test_")}
    for name, fn in tests.items():
        fn()
        print("ok  ", name)
    print(f"{len(tests)} tests passed")
