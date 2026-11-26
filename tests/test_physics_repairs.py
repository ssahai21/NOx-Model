import unittest

import torch

from src.physics_loss import PhysicsLoss, same_day_pair_losses, temporal_decay_loss


class PhysicsRepairTests(unittest.TestCase):
    def pair_losses(self, fluxes, dates=(100, 100), fires=(0, 0)):
        pred = torch.asinh(torch.tensor(fluxes, dtype=torch.float32)).requires_grad_()
        spatial, wind, count = same_day_pair_losses(
            pred, torch.tensor([40., 40.]), torch.tensor([-120., -119.99]),
            torch.ones(2), torch.zeros(2), torch.tensor(dates), torch.tensor(fires),
        )
        return pred, spatial, wind, count

    def test_cross_date_pairs_have_zero_penalty_and_gradient(self):
        pred, spatial, wind, count = self.pair_losses([10., 0.], dates=(100, 101))
        self.assertEqual(count, 0)
        self.assertEqual((spatial + wind).item(), 0)
        gradient, = torch.autograd.grad(spatial + wind, pred)
        torch.testing.assert_close(gradient, torch.zeros_like(pred))

    def test_cross_fire_pairs_are_excluded(self):
        _, spatial, wind, count = self.pair_losses([10., 0.], fires=(0, 1))
        self.assertEqual(count, 0)
        self.assertEqual((spatial + wind).item(), 0)

    def test_downwind_direction_uses_source_to_destination(self):
        pred, _, wind, count = self.pair_losses([10., 0.])
        self.assertEqual(count, 2)
        self.assertAlmostEqual(wind.item(), 5., places=4)
        gradient, = torch.autograd.grad(wind, pred)
        self.assertGreater(gradient[0].item(), 0)
        self.assertLess(gradient[1].item(), 0)
        _, _, reverse_wind, _ = self.pair_losses([0., 10.])
        self.assertEqual(reverse_wind.item(), 0)

    def test_signed_cancellation_does_not_explode_repaired_decay(self):
        pred = torch.tensor([-1., 1.], requires_grad=True)
        days = torch.tensor([100., 100.])
        loss = temporal_decay_loss(pred, days, stable_normalization=True)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertLess(loss.item(), 10)
        gradient, = torch.autograd.grad(loss, pred)
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertGreater(gradient.norm().item(), 0)

    def test_log_compression_preserves_gradient_above_old_cap(self):
        raw = torch.zeros(2, 1)
        gradients = []
        for repaired in [False, True]:
            pred = torch.tensor([[-5.], [-4.]], requires_grad=True)
            criterion = PhysicsLoss(
                lambda_physics=1, lambda_box=0, lambda_deposition=0,
                lambda_decay=0, lambda_spatial=0, lambda_wind=0,
                stable_regularization=repaired,
            )
            _, _, physics, parts = criterion(pred, pred.detach(), raw, ['days_post_fire'])
            self.assertGreater(parts['physics_total'], 1000)
            gradient, = torch.autograd.grad(physics, pred)
            self.assertTrue(torch.isfinite(gradient).all().item())
            gradients.append(gradient.norm().item())
        self.assertEqual(gradients[0], 0)
        self.assertGreater(gradients[1], 0)

    def test_combined_loss_uses_stable_decay_and_disables_ablated_term(self):
        pred = torch.tensor([[-1.], [1.]], requires_grad=True)
        raw = torch.full((2, 1), 100.)
        for weight in [0., 1.]:
            criterion = PhysicsLoss(
                lambda_box=0, lambda_deposition=0, lambda_nonneg=0,
                lambda_spatial=0, lambda_wind=0, lambda_decay=weight,
                stable_regularization=True,
            )
            _, _, loss, parts = criterion(pred, pred.detach(), raw, ['days_post_fire'])
            expected = temporal_decay_loss(pred, raw[:, 0], stable_normalization=True).item() if weight else 0
            self.assertAlmostEqual(parts['decay'], expected, places=5)
            self.assertTrue(torch.isfinite(loss).item())

    def test_repaired_pairs_require_calendar_context(self):
        criterion = PhysicsLoss(same_day_pairs=True)
        with self.assertRaisesRegex(ValueError, 'Missing physical context'):
            criterion(torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 1), ['days_post_fire'])


if __name__ == '__main__':
    unittest.main()
