import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sudoku_grpo import solver, verify

# val puzzle solvable by singles, uses both naked (38) and hidden (6) steps
P = "54..6..1..72.51.466....4..5..4.167.3..943..5..1.5726.4...62943.......9...961....."
S = "548763219972851346631294875254916783769438152813572694185629437427385961396147528"
# test_grid.py fixture: no single anywhere at the start -> solver must stall
HARD = "5...27..9..41......1..5.3...92.6.8...5......66..7..29.8...7...2.......8...9..36.."


class TestSolver(unittest.TestCase):
    def test_solves_and_matches(self):
        ok, steps, final = solver.solve_steps(P)
        self.assertTrue(ok)
        self.assertEqual(final, S)
        self.assertEqual(len(steps), P.count("."))

    def test_formats(self):
        _, steps, _ = solver.solve_steps(P)
        l0, l1, l2 = (solver.trace_text(steps, lv).splitlines()[0] for lv in (0, 1, 2))
        self.assertRegex(l0, r"^r\dc\d=\d$")
        self.assertTrue(l1.startswith(l0 + " "))
        self.assertTrue(l2.startswith(l1))

    def test_stall_on_hard(self):
        ok, steps, _ = solver.solve_steps(HARD)
        self.assertFalse(ok)
        self.assertEqual(steps, [])


class TestVerify(unittest.TestCase):
    def setUp(self):
        _, self.steps, _ = solver.solve_steps(P)

    def test_generated_traces_fully_valid(self):
        for lv in solver.LEVELS:
            chk = verify.check_trace(P, solver.trace_text(self.steps, lv), S)
            s = chk.summary(lv)
            self.assertTrue(s["all_ok"], (lv, s))
            self.assertEqual(chk.final, S)

    def _corrupt(self, lv, fn):
        lines = solver.trace_text(self.steps, lv).splitlines()
        lines[0] = fn(lines[0])
        return verify.check_trace(P, "\n".join(lines), S)

    def test_wrong_digit_illegal_or_incorrect(self):
        chk = self._corrupt(0, lambda l: l[:-1] + str(int(l[-1]) % 9 + 1))
        st = chk.steps[0]
        self.assertFalse(st.ok(0))

    def test_l3_evidence_first(self):
        lines = solver.trace_text(self.steps, 3).splitlines()
        self.assertRegex(lines[0], r"^(r\dc\d r:\d* c:\d* b:\d* => \d|[rcb]\d d\d \[.*\] => r\dc\d)$")
        self.assertTrue(any(l.startswith(("b", "c", "r")) and " d" in l for l in lines))  # hidden form present
        bad = lines[0].rsplit("=> ", 1)
        bad = bad[0] + "=> " + (str(int(bad[1]) % 9 + 1) if bad[1].isdigit() else "r9c9")
        chk = verify.check_trace(P, "\n".join([bad] + lines[1:]), S)
        self.assertFalse(chk.steps[0].ok(3))

    def test_hidden_steps_present_and_valid(self):
        self.assertTrue(any(st.rule == "hidden" for st in self.steps))
        lines = solver.trace_text(self.steps, 2).splitlines()
        i = next(k for k, st in enumerate(self.steps) if st.rule == "hidden")
        chk = verify.check_trace(P, "\n".join(lines[:i + 1]), S)
        self.assertTrue(chk.steps[i].claim and chk.steps[i].evidence)

    def test_false_evidence(self):
        def bump(l):  # change one evidence digit-set by dropping its first char after r:
            return l.replace(" r:", " r:9", 1) if " r:" in l else l.replace("[", "[r1c1:r ", 1)
        chk = self._corrupt(2, bump)
        self.assertFalse(chk.steps[0].evidence)
        self.assertFalse(chk.summary(2)["all_ok"])

    def test_false_rule_claim(self):
        # tag a naked step as hidden in an unrelated unit (or vice versa)
        def swap(l):
            parts = l.split()
            return f"{parts[0]} hidden b9" if parts[1] == "naked" else f"{parts[0]} naked"
        chk = self._corrupt(1, swap)
        self.assertFalse(chk.steps[0].ok(1))

    def test_garbage_line(self):
        chk = verify.check_trace(P, "Wait, let me check\nr1c2=8", S)
        self.assertFalse(chk.steps[0].parsed)
        self.assertTrue(chk.steps[1].parsed)


if __name__ == "__main__":
    unittest.main()
