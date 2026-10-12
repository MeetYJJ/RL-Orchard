import tempfile
import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import SACActor


class ActorTests(unittest.TestCase):
    def test_future_sac_api_shapes_and_bounds(self):
        actor = SACActor()
        state = torch.randn(4, 12)
        mean, log_std = actor(state)
        action = actor.deterministic_action(state)
        self.assertEqual(tuple(mean.shape), (4, 6))
        self.assertEqual(tuple(log_std.shape), (4, 6))
        self.assertEqual(tuple(action.shape), (4, 6))
        self.assertTrue(torch.all(action <= 1.0))
        self.assertTrue(torch.all(action >= -1.0))

    def test_log_std_is_frozen_during_bc(self):
        actor = SACActor()
        before = {key: value.detach().clone()
                  for key, value in actor.log_std_head.state_dict().items()}
        optimizer = torch.optim.Adam(
            [parameter for parameter in actor.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        loss = torch.nn.functional.mse_loss(
            actor.deterministic_action(torch.randn(8, 12)), torch.zeros(8, 6)
        )
        loss.backward()
        optimizer.step()
        for key, value in actor.log_std_head.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
            self.assertFalse(value.requires_grad)

    def test_checkpoint_round_trip(self):
        actor = SACActor()
        state = torch.randn(2, 12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save({
                "actor_state_dict": actor.state_dict(),
                "actor_class": "SACActor",
                "architecture_version": SACActor.architecture_version,
                "state_dim": 12,
                "action_dim": 6,
                "hidden_dim": 256,
            }, str(path))
            restored = SACActor()
            restored.load_state_dict(torch.load(str(path), map_location="cpu")[
                "actor_state_dict"
            ])
            self.assertTrue(torch.allclose(
                actor.deterministic_action(state), restored.deterministic_action(state)
            ))


if __name__ == "__main__":
    unittest.main()
