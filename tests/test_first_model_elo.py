import json
import tempfile
import unittest
from pathlib import Path

from core.elo import run_elo_analysis_eqbench3
from merge_results_to_canonical import find_merge_candidates


class FirstModelEloTests(unittest.TestCase):
    def test_single_model_without_opponents_is_a_successful_noop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            leaderboard_elo = temp_path / "leaderboard_elo.json"
            local_elo = temp_path / "local_elo.json"
            empty_elo = {"__metadata__": {"global_pairwise_comparisons": []}}
            leaderboard_elo.write_text(json.dumps(empty_elo), encoding="utf-8")
            local_elo.write_text(json.dumps(empty_elo), encoding="utf-8")

            model_name = "first-model"
            merged_runs = {
                "run-1": {
                    "model_name": model_name,
                    "scenario_tasks": {
                        "0": {
                            "singleton-regression": {
                                "status": "completed",
                                "conversation_history": [
                                    {"role": "user", "content": "Hello"},
                                    {"role": "assistant", "content": "Hi"},
                                ],
                                "debrief_response": "complete",
                            }
                        }
                    },
                }
            }

            snapshot, error = run_elo_analysis_eqbench3(
                run_key="run-1",
                leaderboard_elo_file=str(leaderboard_elo),
                local_elo_file=str(local_elo),
                merged_runs_data=merged_runs,
                test_model=model_name,
                judge_model="unused-judge",
                api_clients={},
                scenarios_data={},
                concurrency=1,
                recompute_existing=False,
            )

            self.assertIsNone(error)
            self.assertIn(model_name, snapshot)
            self.assertEqual(snapshot[model_name]["elo"], snapshot[model_name]["elo_norm"])
            self.assertIn("sigma", snapshot[model_name])
            self.assertIn("ci_low_norm", snapshot[model_name])
            self.assertIn("ci_high_norm", snapshot[model_name])

            persisted = json.loads(local_elo.read_text(encoding="utf-8"))
            self.assertEqual(persisted[model_name], snapshot[model_name])


class MergeCandidateRecoveryTests(unittest.TestCase):
    @staticmethod
    def _comparison():
        return {
            "scenario_id": "singleton-regression",
            "pair": {
                "test_model": "first-model",
                "neighbor_model": "second-model",
                "iteration_index": 0,
            },
            "fraction_for_test": 0.5,
        }

    def test_known_first_model_error_is_reconciled_from_later_solved_elo(self):
        local_runs = {
            "run-1": {
                "model_name": "first-model",
                "results": {
                    "average_rubric_score": 16.0,
                    "rubric_error": None,
                    "elo_raw": 1000.0,
                    "elo_normalized": 1000.0,
                    "elo_error": (
                        "Final solve/normalization failed: cannot access local "
                        "variable 'rank_window' where it is not associated with a value"
                    ),
                },
            }
        }
        local_elo = {
            "first-model": {"elo": 1112.5, "elo_norm": 1098.25},
            "__metadata__": {"global_pairwise_comparisons": [self._comparison()]},
        }

        candidates = find_merge_candidates(local_runs, local_elo, {}, {})

        self.assertEqual([candidate["model_name"] for candidate in candidates], ["first-model"])
        results = local_runs["run-1"]["results"]
        self.assertIsNone(results["elo_error"])
        self.assertEqual(results["elo_raw"], 1112.5)
        self.assertEqual(results["elo_normalized"], 1098.25)

    def test_unrelated_elo_failure_is_not_silenced(self):
        local_runs = {
            "run-1": {
                "model_name": "first-model",
                "results": {
                    "average_rubric_score": 16.0,
                    "rubric_error": None,
                    "elo_raw": 1000.0,
                    "elo_normalized": 1000.0,
                    "elo_error": "Judge API request failed after retries",
                },
            }
        }
        local_elo = {
            "first-model": {"elo": 1112.5, "elo_norm": 1098.25},
            "__metadata__": {"global_pairwise_comparisons": [self._comparison()]},
        }

        candidates = find_merge_candidates(local_runs, local_elo, {}, {})

        self.assertEqual(candidates, [])
        self.assertEqual(
            local_runs["run-1"]["results"]["elo_error"],
            "Judge API request failed after retries",
        )


if __name__ == "__main__":
    unittest.main()
