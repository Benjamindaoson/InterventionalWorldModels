import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco")

from iwm.data import generate
from iwm.evaluation import evaluate
from iwm.training import train


def test_checkpoint_resume(tmp_path):
    dataset = generate(tmp_path / "data", scenes=5, horizon=2)
    config = {"model": "action_predictor", "dataset": str(dataset),
              "run_dir": str(tmp_path / "first"), "seed": 5,
              "hidden": 8, "epochs": 1, "batch_size": 4}
    train(config)
    first = torch.load(tmp_path / "first" / "last.pt", weights_only=False)
    assert first["epoch"] == 0
    config.update({"run_dir": str(tmp_path / "resumed"), "epochs": 2,
                   "resume": str(tmp_path / "first" / "last.pt")})
    train(config)
    resumed = torch.load(tmp_path / "resumed" / "last.pt", weights_only=False)
    assert resumed["epoch"] == 1
    assert resumed["optimizer"]["state"]
    assert resumed["scheduler"]["last_epoch"] > first["scheduler"]["last_epoch"]
    metrics = evaluate(tmp_path / "resumed" / "last.pt", dataset)
    assert metrics["one_step_mse"] == metrics["teacher_forcing_mse"]
    assert len(metrics["per_scene"]) == 1
