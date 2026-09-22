from ray_policy import cli
from ray_policy.tuning import PolicySearchResult


def test_cli_output(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_tune_search", lambda num_samples: PolicySearchResult(4, 9, 123.45, 6))
    cli.main()
    output = capsys.readouterr().out
    assert "reorder_point: 4" in output
    assert "order_quantity: 9" in output
    assert "validation_cost: 123.45" in output
    assert "trials: 6" in output
