import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sudoku_grpo import grid, prompt

P = "5...27..9..41......1..5.3...92.6.8...5......66..7..29.8...7...2.......8...9..36.."
S = "583427169974136528216859374792364851351298746648715293865971432137642985429583617"


class T(unittest.TestCase):
    def test_solution_valid(self):
        self.assertTrue(grid.is_complete_valid(S))
        self.assertFalse(grid.is_complete_valid(P))
        self.assertTrue(grid.givens_preserved(P, S))

    def test_conflicts(self):
        bad = "55" + S[2:]
        self.assertGreater(grid.conflicts(bad), 0)
        self.assertEqual(grid.conflicts(S), 0)

    def test_cell_accuracy(self):
        self.assertEqual(grid.cell_accuracy(P, S, S), 1.0)
        self.assertEqual(grid.cell_accuracy(P, P.replace(".", "1"), S) < 0.2, True)

    def test_roundtrip_answer(self):
        text = prompt.completion("steps", S)
        self.assertEqual(grid.parse_answer(text), S)
        self.assertIsNone(grid.parse_answer("<answer>123</answer>"))
        self.assertIsNone(grid.parse_answer("no tags"))

    def test_literal(self):
        cells = [[0 if ch == "." else int(ch) for ch in P[r*9:(r+1)*9]] for r in range(9)]
        import json
        self.assertEqual(grid.parse_grid_literal(json.dumps(cells)), P)
        self.assertEqual(grid.normalize(P.replace(".", "0")), P)


if __name__ == "__main__":
    unittest.main()
