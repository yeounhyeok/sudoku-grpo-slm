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

    def test_reason_is_solution_free(self):
        g = gold()
        self.assertAlmostEqual(reward.score(P, S, g)["reason"], 1.0)
        wrong_key = S[::-1]                       # answer key is irrelevant to the reason reward
        self.assertAlmostEqual(reward.score(P, wrong_key, g)["reason"], 1.0)
        self.assertEqual(reward.score(P, wrong_key, g)["exact"], 0.0)

    def test_reason_graded(self):
        lines = gold().split("\n")
        i = next(k for k, l in enumerate(lines) if " naked " in l)
        lines[i] = lines[i].replace(" r:", " r:9", 1)   # one false evidence field -> partial credit
        r = reward.score(P, S, "\n".join(lines))["reason"]
        self.assertAlmostEqual(r, 1 - 0.4 / 3 / P.count("."), places=6)
        self.assertLess(reward.score(P, S, "r1c3=1 naked r:1 c:2 b:3\n</think>")["reason"], 0.02)

    def test_legal_guess_earns_nothing(self):
        # a legal but unforced digit (not a single) scores 0 and is not placed on the board
        from sudoku_grpo import solver as sv, verify as vf
        g = sv.to_cells(P)
        x = next(i for i in range(81) if not g[i] and len(sv.candidates(g, i)) > 1
                 and not any(vf.entailed(g, i, d) for d in sv.candidates(g, i)))
        d = min(sv.candidates(g, x))
        line = f"{sv.cell_name(x)}={d}"
        r = reward.score(P, S, line + "\n</think>\n\n<answer>\n" + format_grid(S) + "\n</answer>", 0)
        self.assertEqual(r["reason"], 0.0)

    def test_unsorted_evidence_accepted(self):
        lines = gold().split("\n")
        i = next(k for k, l in enumerate(lines) if " naked r:" in l and len(l.split("r:")[1].split()[0]) > 1)
        head, tail = lines[i].split(" r:", 1)
        rd, rest = tail.split(" ", 1)
        lines[i] = f"{head} r:{rd[::-1]} {rest}"
        self.assertAlmostEqual(reward.score(P, S, "\n".join(lines))["reason"], 1.0)

    def test_unclosed_reason_zero(self):
        self.assertEqual(reward.score(P, S, gold().split("</think>")[0])["reason"], 0.0)

    def test_answer_must_follow_trace(self):
        bad = S[:-1] + str(int(S[-1]) % 9 + 1)
        c = gold().split("<answer>")[0] + "<answer>\n" + format_grid(bad) + "\n</answer>"
        self.assertEqual(reward.score(P, S, c)["format"], 0.0)

    def test_unclosed(self):
        r = reward.score(P, S, gold().split("</think>")[0])
        self.assertEqual((r["format"], r["exact"]), (0.0, 0.0))

    def test_trl_signature(self):
        f, s, ev, rs, e = reward.make_reward_funcs(2)
        c = [gold(), "garbage"]
        self.assertEqual(f(["p", "p"], c, puzzle=[P, P], solution=[S, S]), [1.0, 0.0])
        self.assertEqual(e(["p", "p"], c, puzzle=[P, P], solution=[S, S]), [1.0, 0.0])
        self.assertEqual(s(["p", "p"], c, puzzle=[P, P], solution=[S, S])[1], 0.0)


if __name__ == "__main__":
    unittest.main()
