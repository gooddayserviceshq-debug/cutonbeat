import unittest
import numpy as np

from cutonbeat.core import beat_grid, energy_tiers, plan_cuts, loudest_windows, snap
from cutonbeat.synth import drum_track


class TestBeatGrid(unittest.TestCase):
    def test_recovers_known_tempo(self):
        for bpm in (92.0, 120.0, 143.0):
            beats, est, kicks = beat_grid(drum_track(bpm, 45))
            self.assertAlmostEqual(est, bpm, delta=1.0, msg=f"{bpm} bpm read as {est:.2f}")

    def test_grid_lands_on_kicks_not_offbeat_hats(self):
        """The synthetic track has broadband hats on every off-beat. Full-band flux alone locks
        the grid onto them; every beat must instead sit within one video frame of a kick."""
        for bpm in (92.0, 120.0, 143.0):
            P = 60 / bpm
            beats, _, kicks = beat_grid(drum_track(bpm, 90))
            off = np.abs(((beats + P / 2) % P) - P / 2)
            self.assertLess(off.max(), 1 / 24, f"{bpm} bpm: beat {off.max() * 1000:.0f} ms from a kick")
            self.assertLess(np.median(np.abs(((kicks + P / 2) % P) - P / 2)), 0.03)

    def test_grid_does_not_drift(self):
        """Last beat of a 3-minute song still lands within 30 ms of a real kick."""
        bpm = 128.0
        beats, est, _ = beat_grid(drum_track(bpm, 180))
        P = 60 / bpm
        err = np.abs(((beats[-5:] + P / 2) % P) - P / 2)
        self.assertLess(err.max(), 1 / 24, f"grid drifted {err.max() * 1000:.0f} ms by the end")


class TestPlanCuts(unittest.TestCase):
    def setUp(self):
        self.y = drum_track(120, 60)
        self.beats, self.bpm, self.kicks = beat_grid(self.y)
        self.tiers, self.energy = energy_tiers(self.y, self.beats)
        self.cuts = plan_cuts(self.beats, self.tiers, 60.0, self.kicks, 60 / self.bpm)

    def test_covers_song_with_no_gaps(self):
        self.assertEqual(self.cuts[0]["start"], 0.0)
        self.assertEqual(self.cuts[-1]["end"], 60.0)
        for a, b in zip(self.cuts, self.cuts[1:]):
            self.assertEqual(a["end"], b["start"])
            self.assertGreater(a["end"], a["start"])

    def test_loud_section_cuts_faster(self):
        """The synthetic track is loud from 24s to 48s: cuts there are shorter than in the intro."""
        length = lambda lo, hi: np.mean([c["end"] - c["start"] for c in self.cuts if lo <= c["start"] < hi])
        self.assertLess(length(26, 46), length(2, 20))

    def test_snap_only_within_tolerance(self):
        kicks = np.array([1.0, 2.0])
        self.assertEqual(snap(1.04, kicks, 0.05), 1.0)
        self.assertEqual(snap(1.20, kicks, 0.05), 1.20)
        self.assertEqual(snap(1.5, np.array([]), 0.05), 1.5)

    def test_shorts_windows_are_loud_bar_aligned_and_disjoint(self):
        win = int(round(10 / (60 / self.bpm)))
        idx = loudest_windows(self.energy, self.beats, win, 2)
        self.assertEqual(len(idx), 2)
        for i in idx:
            self.assertEqual(i % 4, 0)
            self.assertGreaterEqual(self.beats[i], 6.0)
        self.assertGreaterEqual(abs(idx[1] - idx[0]), win)
        self.assertTrue(all(20 <= self.beats[i] <= 50 for i in idx), [self.beats[i] for i in idx])


if __name__ == "__main__":
    unittest.main()
