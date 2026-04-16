"""Stage 9 gate: Habitat (remote GPU) training track.

On hosts *without* habitat-sim (notably the Apple M4 where we develop the
local track), every runtime test auto-skips. Only the static checks always
run: they confirm the Habitat scaffolding *exists and is syntactically valid*
so we notice if it regresses on a laptop commit.

On the remote GPU docker image, habitat-sim imports cleanly and the runtime
tests exercise HabitatNavEnv end-to-end against a small scene asset.
"""
from __future__ import annotations
from pathlib import Path
import importlib
import importlib.util
import os
import pytest


REPO = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------- helpers
def _has_habitat() -> bool:
    return importlib.util.find_spec("habitat_sim") is not None


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_scripts_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # compile + import, no main()
    return mod


# =========================================================================
# Static checks — these MUST pass on every host (M4 included).
# =========================================================================
def test_habitat_env_module_importable_without_habitat_sim():
    """The adapter module imports cleanly even if habitat-sim is not present;
    the error is deferred to class construction. This guards the laptop dev
    loop against accidental hard imports of habitat-sim."""
    mod = importlib.import_module("bev_vawa.envs.habitat_env")
    assert hasattr(mod, "HabitatNavEnv")
    # the module-level flag should exist and be bool
    assert isinstance(getattr(mod, "_HAS_HABITAT"), bool)


def test_habitat_rollout_module_importable_without_habitat_sim():
    mod = importlib.import_module("bev_vawa.data.rollout_habitat")
    assert hasattr(mod, "generate_dataset_habitat")
    assert hasattr(mod, "generate_one_scene")


def test_habitat_scripts_compile():
    """Both CLI scripts must at least compile + import on macOS (habitat-sim
    is imported lazily inside ``main()``)."""
    _load_script("generate_data_habitat")
    _load_script("eval_habitat")


def test_habitat_configs_parse():
    """All configs/habitat/*.yaml must load via our config loader (which
    follows the ``inherit:`` chain)."""
    from bev_vawa.utils import load_config
    for name in ("default.yaml", "hssd.yaml", "procthor.yaml"):
        cfg = load_config(str(REPO / "configs" / "habitat" / name))
        # expected sections inherited from root default
        for key in ("env", "bev", "va", "wa", "fusion", "train", "data"):
            assert key in cfg, f"{name} missing section {key!r}"
        # habitat-specific block
        assert "habitat" in cfg, f"{name} missing 'habitat' block"


def test_dockerfile_and_remote_script_exist():
    df = REPO / "docker" / "habitat.Dockerfile"
    sh = REPO / "docker" / "train_remote.sh"
    md = REPO / "docker" / "README.md"
    assert df.exists() and df.stat().st_size > 500
    assert sh.exists() and sh.stat().st_mode & 0o100, "train_remote.sh must be executable"
    assert md.exists() and md.stat().st_size > 1000

    text = df.read_text()
    assert "habitat-sim" in text
    assert "cuda" in text.lower()

    sh_text = sh.read_text()
    assert "generate_data_habitat" in sh_text
    assert "eval_habitat" in sh_text


def test_habitat_construct_without_install_raises_cleanly():
    """On a host without habitat-sim, constructing HabitatNavEnv must raise
    ImportError with a helpful message — not a cryptic AttributeError."""
    if _has_habitat():
        pytest.skip("habitat-sim is installed; runtime tests cover this path")
    from bev_vawa.envs.habitat_env import HabitatNavEnv
    with pytest.raises(ImportError, match="habitat-sim"):
        HabitatNavEnv(env_cfg={
            "depth_wh": [64, 64], "depth_fov_deg": 90, "depth_max_m": 3.0,
            "control_dt": 0.1, "max_lin_vel": 0.4, "max_ang_vel": 1.2,
            "goal_tol_m": 0.25, "max_collisions": 10, "max_episode_steps": 200,
        }, scene_glb="/nonexistent.glb")


# =========================================================================
# Runtime tests — only on the remote docker image (habitat-sim + a scene).
# =========================================================================
@pytest.mark.skipif(not _has_habitat(), reason="habitat-sim not installed (laptop / non-docker host)")
def test_habitat_env_reset_and_step_smoke():
    """Smoke test on the remote box: construct, reset, take a few steps."""
    scene = os.environ.get("BEV_VAWA_HABITAT_TEST_SCENE")
    if not scene or not Path(scene).exists():
        pytest.skip("set BEV_VAWA_HABITAT_TEST_SCENE to a valid .glb to enable")
    from bev_vawa.envs.habitat_env import HabitatNavEnv
    import numpy as np

    cfg_env = {
        "depth_wh": [64, 64], "depth_fov_deg": 90, "depth_max_m": 3.0,
        "control_dt": 0.1, "max_lin_vel": 0.4, "max_ang_vel": 1.2,
        "goal_tol_m": 0.25, "max_collisions": 10, "max_episode_steps": 50,
    }
    env = HabitatNavEnv(cfg_env, scene_glb=scene, seed=0)
    try:
        obs = env.reset(seed=42)
        assert obs["depth"].shape == (64, 64)
        assert 0.0 <= float(obs["depth"].min()) <= float(obs["depth"].max()) <= 3.0 + 1e-3
        assert obs["goal_vec"].shape == (2,)
        for _ in range(5):
            step = env.step(np.array([0.2, 0.1]))
            assert step.obs["depth"].shape == (64, 64)
            if step.done:
                break
    finally:
        env.close()
