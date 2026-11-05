import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sudoku_grpo import reward, solver
from sudoku_grpo.grid import format_grid

P = "54..6..1..72.51.466....4..5..4.167.3..943..5..1.5726.4...62943.......9...961....."
S = "548763219972851346631294875254916783769438152813572694185629437427385961396147528"


def gold(level=2):
    _, steps, _ = solver.solve_steps(P)
    return solver.trace_text(steps, level) + "\n</think>\n\n<answer>\n" + format_grid(S) + "\n</answer>"


class TestReward(unittest.TestCase):
    def test_gold_full_marks(self):
        for lv in (0, 2, 3):
            r = reward.score(P, S, gold(lv), lv)
            self.assertEqual((r["format"], r["step"], r["exact"]), (1.0, 1.0, 1.0), lv)

    def test_first_error_cuts_step_reward(self):
        lines = gold().split("\n")
        k = 10
        a, b = lines[k].split("=", 1)
        lines[k] = a + "=" + str(int(b[0]) % 9 + 1) + b[1:]
        r = reward.score(P, S, "\n".join(lines))
        self.assertAlmostEqual(r["step"], k / P.count("."))
        self.assertEqual(r["exact"], 1.0)  # answer block untouched

    def test_evidence_separate(self):
        lines = gold().split("\n")
        lines[0] = lines[0].replace(" r:", " r:9", 1) if " r:" in lines[0] else lines[0].replace("[", "[r1c1:r ", 1)
        r = reward.score(P, S, "\n".join(lines))
        self.assertEqual(r["step"], 1.0)          # placements still all correct
        self.assertLess(r["evidence"], 1.0)        # one false justification

    def test_unclosed(self):
        r = reward.score(P, S, gold().split("</think>")[0])
        self.assertEqual((r["format"], r["exact"]), (0.0, 0.0))

    def test_trl_signature(self):
        f, s, ev, e = reward.make_reward_funcs(2)
        c = [gold(), "garbage"]
        self.assertEqual(f(["p", "p"], c, puzzle=[P, P], solution=[S, S]), [1.0, 0.0])
        self.assertEqual(e(["p", "p"], c, puzzle=[P, P], solution=[S, S]), [1.0, 0.0])
        self.assertEqual(s(["p", "p"], c, puzzle=[P, P], solution=[S, S])[1], 0.0)


if __name__ == "__main__":
    unittest.main()
