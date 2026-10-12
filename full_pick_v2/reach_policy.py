"""Frozen BC/DAgger ReachActor inference, with unchanged normalization."""
from __future__ import annotations
import hashlib
import json
import importlib.util
import numpy as np
from .config import BUNDLE, CHECKPOINT, NORMALIZATION

ACTION_BOUNDS = np.asarray([.0015]*3 + [.0075]*3, dtype=np.float64)

class ReachPolicy:
    def __init__(self):
        import torch
        torch.set_num_threads(1)
        # Load the actual deployed architecture by path, without sys.path/package
        # side effects. This is model code, never a legacy robot state machine.
        spec = importlib.util.spec_from_file_location("_full_pick_frozen_actor", BUNDLE / "training/models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.torch = torch
        manifest = json.loads((BUNDLE / "shadow_bundle_manifest.json").read_text())["files"]
        for path in (CHECKPOINT, NORMALIZATION, BUNDLE / "training/models.py"):
            key = str(path.relative_to(BUNDLE))
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]["sha256"]:
                raise ValueError(f"frozen artifact checksum mismatch: {path}")
        cp = torch.load(str(CHECKPOINT), map_location="cpu")
        if (int(cp["state_dim"]), int(cp["action_dim"]), int(cp["hidden_dim"])) != (15, 6, 256):
            raise ValueError("ReachActor architecture mismatch")
        self.actor = module.SACActor(15, 6, 256)
        self.actor.load_state_dict(cp["model_state_dict"], strict=True)
        self.actor.eval()
        with np.load(NORMALIZATION, allow_pickle=False) as z:
            self.mean = np.asarray(z["state_mean"], dtype=np.float64)
            self.std = np.asarray(z["state_std"], dtype=np.float64)
            if int(z["state_dim"]) != 15 or int(z["action_dim"]) != 6:
                raise ValueError("normalization dimensions mismatch")
            if not np.allclose(z["action_high"], ACTION_BOUNDS, atol=1e-9, rtol=0) or not np.allclose(z["action_low"], -ACTION_BOUNDS, atol=1e-9, rtol=0):
                raise ValueError("normalization action limits mismatch")
        if self.mean.shape != (15,) or self.std.shape != (15,) or not np.all(np.isfinite(self.mean)) or not np.all(np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError("invalid frozen normalization")

    def infer(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.shape != (15,) or not np.all(np.isfinite(raw)):
            raise ValueError("OBSERVATION_INVALID")
        normalized = (raw - self.mean) / self.std
        with self.torch.no_grad():
            result = self.actor.deterministic_action(self.torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0))[0].numpy().astype(np.float64)
        if result.shape != (6,) or not np.all(np.isfinite(result)) or np.any(np.abs(result) > 1 + 1e-7):
            raise ValueError("ACTOR_INVALID")
        return result * ACTION_BOUNDS
