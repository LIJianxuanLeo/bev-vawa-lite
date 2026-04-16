"""Stage 0 gate: deps import, MuJoCo renders a depth frame, MPS conv works."""
from __future__ import annotations
import numpy as np


def test_imports():
    import torch, mujoco, yaml, tqdm  # noqa: F401
    from bev_vawa.utils import get_device, set_seed, load_config, get_logger  # noqa


def test_mujoco_depth_smoke():
    import mujoco
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3"/>
        <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
        <geom name="box" type="box" pos="0 1 0.5" size="0.5 0.5 0.5" rgba="0.2 0.4 0.8 1"/>
        <camera name="cam" pos="0 0 0.3" xyaxes="1 0 0 0 0 1"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    r = mujoco.Renderer(model, height=64, width=64)
    r.enable_depth_rendering()
    r.update_scene(data, camera="cam")
    depth = r.render()
    assert depth.shape == (64, 64)
    assert np.isfinite(depth).all()
    assert depth.min() >= 0.0


def test_mps_conv():
    import torch
    from bev_vawa.utils import get_device
    dev = get_device()
    x = torch.randn(2, 3, 8, 8, device=dev)
    conv = torch.nn.Conv2d(3, 4, 1).to(dev)
    y = conv(x)
    assert y.shape == (2, 4, 8, 8)
    assert torch.isfinite(y).all()


def test_config_load(tmp_path):
    import yaml
    from bev_vawa.utils import load_config
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"a": 1, "b": {"c": 2}}))
    cfg = load_config(p, overrides={"b": {"d": 3}})
    assert cfg["a"] == 1 and cfg["b"]["c"] == 2 and cfg["b"]["d"] == 3


def test_seed_determinism():
    import numpy as np
    from bev_vawa.utils import set_seed
    set_seed(123)
    a = np.random.rand(4)
    set_seed(123)
    b = np.random.rand(4)
    assert np.allclose(a, b)
