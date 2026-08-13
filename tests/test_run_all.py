from argparse import Namespace
from pathlib import Path

from experiments.run_all import _command, _paths
from experiments.exp3_cost_accuracy import _default_design_path


def test_runner_defaults_to_quick_and_wires_shared_checkpoint(tmp_path):
    paths = _paths(tmp_path)
    args = Namespace(
        device="cpu",
        dtype="float64",
        seed=7,
        full=False,
        checkpoint=None,
        design_results=None,
    )
    command = _command("exp5", paths, args)
    assert "--quick" in command
    checkpoint_index = command.index("--checkpoint") + 1
    assert (
        Path(command[checkpoint_index])
        == paths["exp1"] / "checkpoints" / "base_model.pt"
    )


def test_plot_command_does_not_request_quick_or_full(tmp_path):
    paths = _paths(tmp_path)
    args = Namespace(device="cpu", dtype="float64", seed=7, full=False)
    command = _command("exp3", paths, args, plots_only=True)
    assert "--plots-only" in command
    assert "--quick" not in command
    assert "--full" not in command


def test_exp5_resume_flag_is_forwarded(tmp_path):
    paths = _paths(tmp_path)
    args = Namespace(
        device="cpu",
        dtype="float64",
        seed=7,
        full=True,
        checkpoint=None,
        design_results=None,
        resume=True,
    )
    assert "--resume" in _command("exp5", paths, args)


def test_direct_exp3_default_design_paths_match_runner_output_roots():
    assert _default_design_path(True) == (
        "outputs/current_paper_quick/design/data/milp_results.json"
    )
    assert _default_design_path(False) == (
        "outputs/current_paper_full/design/data/milp_results.json"
    )
