"""Stage 12 gate: Gibson PointNav v2 track (remote-only runtime; static checks
run everywhere).

Static portion (always runs):
  * ``configs/habitat/gibson{,_occonly}.yaml`` parse via the inherit chain.
  * ``bev_vawa.data.gibson_episodes`` importable, parses the v2 JSON.gz layout
    (flat + ``content/``) on a synthetic pack.
  * ``NavShardDataset`` schema-v2 detection on a synthesised v2 shard.
  * ``scripts/generate_data_habitat.py`` / ``scripts/eval_habitat.py`` compile.
  * ``docker/train_remote.sh`` runs the Gibson pipeline.

Runtime portion (habitat-sim required):
  * End-to-end single-episode rollout on a test scene (skipped unless
    ``BEV_VAWA_GIBSON_TEST_SCENE`` + ``BEV_VAWA_GIBSON_TEST_EPISODES`` are set).
"""
from __future__ import annotations
import gzip
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pytest


REPO = Path(__file__).resolve().parent.parent


def _has_habitat() -> bool:
    return importlib.util.find_spec("habitat_sim") is not None


# =========================================================================
# Static checks
# =========================================================================
def test_gibson_configs_parse_via_inherit_chain():
    from bev_vawa.utils import load_config
    cfg = load_config(str(REPO / "configs" / "habitat" / "gibson.yaml"))
    assert cfg["dataset"]["name"] == "gibson"
    assert cfg["wa"]["enable_dyn"] is True
    assert cfg["wa"]["enable_deadend"] is True
    assert cfg["fusion"]["eta"] == pytest.approx(1.0)
    assert list(cfg["bev"]["channels_enabled"]) == [1, 1, 1]

    cfg_abl = load_config(str(REPO / "configs" / "habitat" / "gibson_occonly.yaml"))
    assert list(cfg_abl["bev"]["channels_enabled"]) == [1, 0, 0]


def _write_json_gz(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(path), "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def test_gibson_episode_loader_parses_flat_layout(tmp_path):
    from bev_vawa.data.gibson_episodes import iter_episodes
    episode_dir = tmp_path / "pointnav" / "gibson" / "v2"
    pack = {
        "episodes": [
            {
                "episode_id": "0",
                "scene_id": "gibson/Allensville.glb",
                "start_position": [0.0, 0.0, 0.0],
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
                "goals": [{"position": [1.5, 0.0, 2.0]}],
                "info": {"geodesic_distance": 3.2},
            },
            {
                "episode_id": "1",
                "scene_id": "gibson/Beechwood.glb",
                "start_position": [1.0, 0.0, 0.0],
                "goals": [{"position": [0.0, 0.0, 2.0]}],
                "info": {"geodesic_distance": 2.1},
            },
        ]
    }
    _write_json_gz(episode_dir / "train" / "train.json.gz", pack)

    eps = list(iter_episodes(str(episode_dir), split="train"))
    assert len(eps) == 2
    assert eps[0]["scene_id"].endswith("Allensville.glb")
    assert eps[0]["start_position"].shape == (3,)
    assert eps[0]["goal_position"].shape == (3,)
    assert eps[0]["geodesic_distance"] == pytest.approx(3.2)
    filt = list(iter_episodes(str(episode_dir), split="train",
                              scene_filter=["Beechwood"]))
    assert len(filt) == 1
    assert "Beechwood" in filt[0]["scene_id"]


def test_gibson_episode_loader_parses_content_layout(tmp_path):
    """PointNav v2 val split typically ships under ``{split}/content/*.json.gz``."""
    from bev_vawa.data.gibson_episodes import iter_episodes
    episode_dir = tmp_path / "pointnav" / "gibson" / "v2"
    for i, scene in enumerate(("Allensville", "Beechwood")):
        pack = {
            "episodes": [{
                "episode_id": str(i),
                "scene_id": f"gibson/{scene}.glb",
                "start_position": [float(i), 0.0, 0.0],
                "goals": [{"position": [float(i) + 1, 0.0, 2.0]}],
                "info": {"geodesic_distance": 1.0 + i},
            }]
        }
        _write_json_gz(episode_dir / "val" / "content" / f"{scene}.json.gz", pack)
    eps = list(iter_episodes(str(episode_dir), split="val"))
    assert len(eps) == 2


def test_gibson_episode_loader_raises_on_empty_dir(tmp_path):
    from bev_vawa.data.gibson_episodes import iter_episodes
    with pytest.raises(FileNotFoundError):
        list(iter_episodes(str(tmp_path), split="train"))


def test_nav_shard_dataset_detects_v2_schema(tmp_path):
    """Write a synthetic v2 shard and verify the loader surfaces the extra keys."""
    from bev_vawa.data.dataset import NavShardDataset

    N, K, H_im, W_im, H_fut = 3, 5, 16, 16, 3
    shard = {
        "depth": np.random.rand(N, H_im, W_im).astype(np.float32) * 2.0,
        "goal": np.random.randn(N, 2).astype(np.float32),
        "expert_wp": np.random.randn(N, 2).astype(np.float32),
        "cand_collision": np.random.randint(0, 2, (N, K)).astype(np.float32),
        "cand_progress": np.random.rand(N, K).astype(np.float32),
        "best_k": np.random.randint(0, K, (N,)).astype(np.int64),
        "anchors": np.random.randn(K, 2).astype(np.float32),
        # v2 keys
        "schema_version": np.asarray(2),
        "future_depth": np.random.rand(N, H_fut, H_im, W_im).astype(np.float32) * 2.0,
        "future_goal": np.random.randn(N, H_fut, 2).astype(np.float32),
        "cand_deadend": np.random.randint(0, 2, (N, K)).astype(np.float32),
    }
    shard_path = tmp_path / "scene_000.npz"
    np.savez(shard_path, **shard)

    ds = NavShardDataset(str(tmp_path), depth_max=3.0)
    assert len(ds) == N
    assert ds.has_future is True
    assert ds.has_deadend is True

    sample = ds[0]
    assert "future_depth" in sample
    assert sample["future_depth"].shape == (H_fut, 1, H_im, W_im)
    assert "future_goal" in sample
    assert sample["future_goal"].shape == (H_fut, 2)
    assert sample["cand_deadend"].shape == (K,)


def test_nav_shard_dataset_v1_still_works(tmp_path):
    """A legacy shard (no schema_version) must still load — the new v2 keys
    simply aren't surfaced, and the WA loss degrades to L_risk + L_prog."""
    from bev_vawa.data.dataset import NavShardDataset

    N, K = 2, 5
    shard = {
        "depth": np.random.rand(N, 16, 16).astype(np.float32) * 2.0,
        "goal": np.random.randn(N, 2).astype(np.float32),
        "expert_wp": np.random.randn(N, 2).astype(np.float32),
        "cand_collision": np.random.randint(0, 2, (N, K)).astype(np.float32),
        "cand_progress": np.random.rand(N, K).astype(np.float32),
        "best_k": np.random.randint(0, K, (N,)).astype(np.int64),
        "anchors": np.random.randn(K, 2).astype(np.float32),
    }
    np.savez(tmp_path / "room_000.npz", **shard)
    ds = NavShardDataset(str(tmp_path), depth_max=3.0)
    assert ds.has_future is False
    assert ds.has_deadend is False
    sample = ds[0]
    assert "future_depth" not in sample
    assert "cand_deadend" not in sample


def test_generate_data_habitat_has_gibson_cli_flags():
    src = (REPO / "scripts" / "generate_data_habitat.py").read_text()
    for token in ("--scene-dir", "--episode-dir", "--split", "generate_dataset_gibson"):
        assert token in src, f"generate_data_habitat.py missing CLI token {token!r}"


def test_eval_habitat_registers_bev_vawa():
    src_eval = (REPO / "scripts" / "eval_habitat.py").read_text()
    assert '"bev_vawa"' in src_eval, "eval_habitat.py missing 'bev_vawa' MODEL_MAP entry"
    assert "BEVVAWA" in src_eval, "eval_habitat.py missing BEVVAWA import"


def test_train_stages_use_common_wa_loss():
    """Stage B/C trainers must route through ``_common.wa_loss_for_stage`` so
    the dynamics + dead-end loss terms participate when future frames are
    available in the batch."""
    for name in ("stage_b_wa", "stage_c_joint"):
        src = (REPO / "bev_vawa" / "train" / f"{name}.py").read_text()
        assert "wa_loss_for_stage" in src, f"{name}.py does not route through wa_loss_for_stage"
    common = (REPO / "bev_vawa" / "train" / "_common.py").read_text()
    assert "encode_future" in common
    assert "lambda_dyn" in common and "lambda_deadend" in common


def test_train_remote_sh_runs_gibson_pipeline():
    sh = (REPO / "docker" / "train_remote.sh").read_text()
    assert "configs/habitat/gibson.yaml" in sh
    assert "pointnav/gibson/v2" in sh
    # the three training stages + eval
    for stage in ("--stage a", "--stage b", "--stage c", "eval_habitat"):
        assert stage in sh, f"train_remote.sh missing {stage!r}"


def test_rollout_habitat_emits_schema_v2():
    src = (REPO / "bev_vawa" / "data" / "rollout_habitat.py").read_text()
    assert "SHARD_SCHEMA_VERSION" in src
    for k in ("future_depth", "cand_deadend", "generate_dataset_gibson"):
        assert k in src, f"rollout_habitat.py missing {k!r}"


# =========================================================================
# Runtime tests — only on the remote docker image with Gibson assets.
# =========================================================================
@pytest.mark.skipif(not _has_habitat(), reason="habitat-sim not installed")
def test_gibson_rollout_smoke():
    scene = os.environ.get("BEV_VAWA_GIBSON_TEST_SCENE")
    episodes = os.environ.get("BEV_VAWA_GIBSON_TEST_EPISODES")
    if not (scene and episodes and Path(scene).exists() and Path(episodes).exists()):
        pytest.skip(
            "set BEV_VAWA_GIBSON_TEST_SCENE + BEV_VAWA_GIBSON_TEST_EPISODES "
            "to enable the Gibson runtime smoke test"
        )
    from bev_vawa.data.rollout_habitat import generate_dataset_gibson
    from bev_vawa.utils import load_config
    cfg = load_config(str(REPO / "configs" / "habitat" / "gibson.yaml"))
    out_dir = Path(os.environ.get("BEV_VAWA_GIBSON_TEST_OUT", "/tmp/gibson_smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = str(Path(scene).parent)
    generate_dataset_gibson(
        cfg=cfg,
        scene_dir=scene_dir,
        episode_dir=episodes,
        split="val",
        out_dir=str(out_dir),
        max_episodes_per_scene=1,
        scene_limit=1,
    )
    shards = list(out_dir.glob("scene_*.npz"))
    assert shards, "no Gibson shards emitted"
    z = np.load(shards[0])
    assert int(z["schema_version"].item()) == 2
    assert "future_depth" in z.files
    assert "cand_deadend" in z.files
