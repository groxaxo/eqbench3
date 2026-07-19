# File: ai/eqbench3/utils/merge_candidates.py

import os
import sys

# Add the parent directory (ai/eqbench3) to sys.path to allow imports like 'from utils. ...'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


import logging

import copy
from datetime import datetime, timezone
import argparse
from collections import defaultdict
from core.elo import (
    filter_comparisons_for_solver,
    models_in_comparisons,
)
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Set, Tuple
from core.trueskill_solver import solve_with_trueskill, normalize_elo_scores
from core.pairwise_judging import _recompute_comparison_stats
# from core.elo import models_in_comparisons           # helper already written # Redundant import
from core.elo_config import DEFAULT_ELO, WIN_MARGIN_BIN_SIZE, WIN_MARGIN_BIN_SIZE_FOR_CI
from utils.file_io import load_json_file, save_json_file
import utils.constants as C
import uuid # For atomic save
from utils.merge_utils import (
    merge_data,
    unmerge_data,
    _recalculate_elo_ratings,
    _atomic_multi_save,
)

# --- Logging Setup ---
def setup_merge_logging(level_str):
    log_level = getattr(logging, level_str.upper(), logging.INFO)

    # Re-initialise logging even if it was configured earlier
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True              # <-- key line
    )

    # If other modules grabbed the root logger before this, make sure they honour the new level.
    logging.getLogger().setLevel(log_level)

    logging.debug(f"Logging level set to {level_str.upper()}")
# --- End Logging Setup ---


# --- End Imports ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Configured in setup_merge_logging

# =========================================================================
# Merge Functionality
# =========================================================================

def _is_valid_elo_score(value: Any) -> bool:
    """Return whether *value* represents a usable numeric ELO score."""
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"skipped", "error", "n/a", "none"}:
            return False
        try:
            float(text)
        except ValueError:
            return False
        return True
    return False


def _is_known_stale_first_model_error(error: Any) -> bool:
    """Identify errors produced solely by the historical no-op first-model path."""
    if not error:
        return False
    message = str(error).lower()
    return (
        "rank_window" in message
        or "no valid comparisons for final solve" in message
    )


def _model_has_matchups(model_name: str, local_elo: Dict[str, Any]) -> bool:
    """Return True when local ELO contains a real pairwise comparison for a model."""
    local_comps = local_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    for comp in local_comps:
        pair = comp.get("pair", {})
        if model_name in (pair.get("test_model"), pair.get("neighbor_model")):
            return True
    return False


def _reconcile_stale_first_model_result(
    model_name: str,
    results: Dict[str, Any],
    local_elo_entry: Dict[str, Any],
    has_matchups: bool,
) -> None:
    """
    Clear only the known first-model artifact after a later successful solve.

    The local ELO entry plus at least one matchup is the authoritative evidence
    that this model was subsequently solved. Unrelated judge/API failures remain
    untouched and continue to block merging.
    """
    if not has_matchups or not _is_known_stale_first_model_error(results.get("elo_error")):
        return

    authoritative_raw = local_elo_entry.get("elo")
    if not _is_valid_elo_score(authoritative_raw):
        return

    authoritative_norm = local_elo_entry.get("elo_norm", authoritative_raw)
    if not _is_valid_elo_score(authoritative_norm):
        authoritative_norm = authoritative_raw

    logging.info(
        "Clearing stale first-model ELO error for %s using the later solved local ELO entry.",
        model_name,
    )
    results["elo_raw"] = authoritative_raw
    results["elo_normalized"] = authoritative_norm
    results["elo_error"] = None


def find_merge_candidates(local_runs, local_elo, canonical_runs, canonical_elo):
    """Identifies runs in local files that meet merge criteria."""
    candidates = []
    processed_model_names = set() # Track models already added as candidates

    # Build set of model names already in canonical data (runs or elo)
    canonical_model_names = set(k for k in canonical_elo if k != "__metadata__")
    for run_data in canonical_runs.values():
        if isinstance(run_data, dict):
            model_name = run_data.get("model_name", run_data.get("test_model"))
            if model_name:
                canonical_model_names.add(model_name)

    logging.info(f"Found {len(canonical_model_names)} unique model names in canonical data.")
    logging.debug(f"Total items found in local_runs: {len(local_runs)}")

    for run_key, run_data in local_runs.items():
        if run_key == "__metadata__": continue # Skip metadata entry
        logging.debug(f"Processing local run key: '{run_key}'")

        if not isinstance(run_data, dict):
            logging.debug(f"Skipping '{run_key}': Run data is not a dictionary.")
            continue

        model_name = run_data.get("model_name", run_data.get("test_model"))
        if not model_name:
            logging.warning(f"Skipping run {run_key}: Missing model name.")
            continue

        # Avoid adding the same model multiple times if it has multiple local runs
        if model_name in processed_model_names:
            logging.debug(f"Skipping '{run_key}' (model '{model_name}'): Model name already processed from a previous run key.")
            continue

        # --- Check Criteria ---
        # 1. Exists in local ELO?
        local_elo_entry = local_elo.get(model_name)
        if not isinstance(local_elo_entry, dict):
            logging.debug(f"Skipping {model_name} ({run_key}): Missing or invalid entry in local ELO file.")
            continue

        # A model is mergeable only after at least one real pairwise matchup.
        has_matchups = _model_has_matchups(model_name, local_elo)
        if not has_matchups:
            logging.debug(f"Skipping {model_name} ({run_key}): No matchups found in local ELO comparisons.")
            continue

        # 2. Completeness (Rubric and ELO scores).
        results = run_data.get("results")
        if not isinstance(results, dict):
            results = {}
            run_data["results"] = results

        _reconcile_stale_first_model_result(
            model_name,
            results,
            local_elo_entry,
            has_matchups,
        )

        rubric_score = results.get("average_rubric_score")
        elo_raw = results.get("elo_raw")
        has_rubric = rubric_score is not None and rubric_score != "Skipped" and results.get("rubric_error") is None
        has_elo = _is_valid_elo_score(elo_raw) and results.get("elo_error") is None

        if not has_rubric:
            logging.debug(f"Skipping {model_name} ({run_key}): Missing valid Rubric score.")
            continue
        if not has_elo:
            logging.debug(f"Skipping {model_name} ({run_key}): Missing valid ELO score.")
            continue

        # 3. No Name Collision in Canonical Data
        if model_name in canonical_model_names:
            logging.debug(f"Skipping {model_name} ({run_key}): Name already exists in canonical data.")
            continue

        # --- Candidate Found ---
        candidates.append({
            "run_key": run_key,
            "model_name": model_name,
            "rubric_score": rubric_score * 5.0 if isinstance(rubric_score, (int, float)) else "N/A",
            "elo_norm": results.get("elo_normalized", "N/A")
        })
        processed_model_names.add(model_name)
        logging.info(f"Found potential merge candidate: {model_name} (from run {run_key})")

    return candidates

# =========================================================================
# Unmerge Functionality
# =========================================================================

def find_unmerge_candidates(canonical_runs, canonical_elo):
    """Identifies models present in canonical files that can be moved to local."""
    candidates = []
    model_names_in_elo = set(k for k in canonical_elo if k != "__metadata__")
    model_names_in_runs = set()
    run_key_map = defaultdict(list) # model_name -> list of run_keys

    for run_key, run_data in canonical_runs.items():
        if run_key == "__metadata__": continue
        if isinstance(run_data, dict):
            model_name = run_data.get("model_name", run_data.get("test_model"))
            if model_name:
                model_names_in_runs.add(model_name)
                run_key_map[model_name].append(run_key)

    # Consider models present in either file for unmerging
    all_canonical_models = sorted(list(model_names_in_elo | model_names_in_runs))

    logging.info(f"Found {len(all_canonical_models)} unique model names in canonical data for potential unmerging.")

    for model_name in all_canonical_models:
        elo_data = canonical_elo.get(model_name, {})
        elo_norm = elo_data.get("elo_norm", "N/A") if isinstance(elo_data, dict) else "N/A"
        run_keys = run_key_map.get(model_name, ["N/A"])

        candidates.append({
            "model_name": model_name,
            "elo_norm": elo_norm,
            "run_keys": run_keys # Store all associated run keys
        })

    return candidates

# =========================================================================
# Delete Functionality
# =========================================================================

def find_delete_candidates(canonical_runs, canonical_elo):
    """Identifies models present in canonical files that can be deleted."""
    candidates = []
    model_names_in_elo = set(k for k in canonical_elo if k != "__metadata__")
    model_names_in_runs = set()
    run_key_map = defaultdict(list) # model_name -> list of run_keys

    for run_key, run_data in canonical_runs.items():
        if run_key == "__metadata__": continue
        if isinstance(run_data, dict):
            model_name = run_data.get("model_name", run_data.get("test_model"))
            if model_name:
                model_names_in_runs.add(model_name)
                run_key_map[model_name].append(run_key)

    # Consider models present in either file for deletion
    all_canonical_models = sorted(list(model_names_in_elo | model_names_in_runs))

    logging.info(f"Found {len(all_canonical_models)} unique model names in canonical data for potential deletion.")

    for model_name in all_canonical_models:
        elo_data = canonical_elo.get(model_name, {})
        elo_norm = elo_data.get("elo_norm", "N/A") if isinstance(elo_data, dict) else "N/A"
        # Find an associated run key (just for info, not strictly needed for deletion logic)
        run_key_example = run_key_map.get(model_name, ["N/A"])[0] # Just show the first one

        candidates.append({
            "model_name": model_name,
            "elo_norm": elo_norm,
            "run_key_example": run_key_example # For display purposes
        })

    return candidates


def delete_data(selected_models_to_delete, canonical_runs, canonical_elo):
    """Removes selected models and their associated data from canonical files."""
    if not selected_models_to_delete:
        return False # Indicate nothing was deleted

    deleted_model_names = set(c['model_name'] for c in selected_models_to_delete)
    logging.info(f"Preparing to delete {len(deleted_model_names)} models: {', '.join(deleted_model_names)}")

    # --- Remove ELO Entries ---
    deleted_elo_count = 0
    for model_name in deleted_model_names:
        if model_name in canonical_elo:
            if model_name != "__metadata__": # Safety check
                del canonical_elo[model_name]
                logging.info(f"Removed ELO entry for {model_name} from canonical ELO.")
                deleted_elo_count += 1
        else:
            logging.warning(f"Model name {model_name} not found in canonical ELO data during delete.")
    logging.info(f"Removed {deleted_elo_count} ELO entries.")

    # --- Remove Associated Run Data ---
    run_keys_to_delete = set()
    for run_key, run_data in canonical_runs.items():
        if run_key == "__metadata__":
            continue
        if isinstance(run_data, dict):
            model_name = run_data.get("model_name", run_data.get("test_model"))
            if model_name in deleted_model_names:
                run_keys_to_delete.add(run_key)

    deleted_run_count = 0
    for run_key in run_keys_to_delete:
        if run_key in canonical_runs:
            del canonical_runs[run_key]
            logging.info(f"Removed run data for {run_key} (model in {deleted_model_names}) from canonical runs.")
            deleted_run_count += 1
    logging.info(f"Removed {deleted_run_count} run entries.")


    # --- Remove Comparisons Involving Deleted Models ---
    if "__metadata__" in canonical_elo and "global_pairwise_comparisons" in canonical_elo["__metadata__"]:
        original_comps = canonical_elo["__metadata__"]["global_pairwise_comparisons"]
        comps_to_keep = []
        removed_comp_count = 0
        for comp in original_comps:
            pair = comp.get("pair", {})
            model_a = pair.get("test_model")
            model_b = pair.get("neighbor_model")
            if model_a not in deleted_model_names and model_b not in deleted_model_names:
                comps_to_keep.append(comp)
            else:
                removed_comp_count += 1
                logging.debug(f"Removing comparison involving deleted model: {model_a} vs {model_b}")

        canonical_elo["__metadata__"]["global_pairwise_comparisons"] = comps_to_keep
        logging.info(f"Removed {removed_comp_count} comparisons involving deleted models from canonical ELO.")
    else:
        logging.info("No comparisons found in canonical ELO metadata to filter.")

    return True # Indicate deletion occurred

# =========================================================================
# Common Functionality (Selection, ELO Recalc, Saving)
# =========================================================================

def select_models_from_list(candidates: List[Dict[str, Any]], action: str) -> List[Dict[str, Any]]:
    """Prompts the user to select models from a list for a given action (merge/delete/unmerge)."""
    if not candidates:
        return []

    print(f"\n--- Candidates for {action.capitalize()} ---")
    if action == "merge":
        for i, cand in enumerate(candidates):
             rubric_str = f"{cand['rubric_score']:.1f}" if isinstance(cand['rubric_score'], (int, float)) else cand['rubric_score']
             print(f"{i+1: >3}. {cand['model_name']} (Run: {cand['run_key']}, Rubric: {rubric_str}, ELO Norm: {cand['elo_norm']})")
    elif action == "delete":
         for i, cand in enumerate(candidates):
             print(f"{i+1: >3}. {cand['model_name']} (ELO Norm: {cand['elo_norm']}, Example Run: {cand.get('run_key_example', 'N/A')})")
    elif action == "unmerge":
         for i, cand in enumerate(candidates):
             run_keys_str = ', '.join(cand.get('run_keys', ['N/A']))
             print(f"{i+1: >3}. {cand['model_name']} (ELO Norm: {cand['elo_norm']}, Run(s): {run_keys_str})")
    else:
        logging.error(f"Unknown action '{action}' in select_models_from_list")
        return [] # Should not happen
    print("----------------------")

    while True:
        try:
            prompt = f"Enter numbers of models to {action} (e.g., 1,3,4), 'all', or 'none': "
            selection = input(prompt).strip().lower()
            if selection == 'none':
                return []
            if selection == 'all':
                return candidates # Return all candidate dicts

            selected_indices = set()
            parts = selection.split(',')
            for part in parts:
                part = part.strip()
                if not part: continue
                index = int(part) - 1
                if 0 <= index < len(candidates):
                    selected_indices.add(index)
                else:
                    print(f"Invalid number: {part}. Please enter numbers between 1 and {len(candidates)}.")
                    raise ValueError("Invalid index")

            # Return the selected candidate dicts
            return [candidates[i] for i in sorted(list(selected_indices))]

        except ValueError:
            print("Invalid input. Please use the specified format.")
        except Exception as e:
            print(f"An error occurred: {e}")




# ─────────────────────────────────────────────────────────────────────
# Recalc Action Specific Logic (Kept separate as it only saves canonical)
# ─────────────────────────────────────────────────────────────────────
def _refresh_comparison_fields(comps: List[Dict[str, Any]]) -> None:
    """Populate margin / fraction fields so every record is usable."""
    changed = 0
    for c in comps:
        if "error" not in c and "judge_response" in c:
            before = c.get("fraction_for_test")
            _recompute_comparison_stats(c)
            if c.get("fraction_for_test") != before:
                changed += 1
    logging.info(f"[recalc] refreshed stats for {changed} comparisons")

# ─────────────────────────────────────────────────────────────────────
def _dual_solve(all_models: Set[str],
                comps: List[Dict[str, Any]]
               ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Dual solve: μ with bin_size=20, σ with bin_size for CI."""

    filtered = filter_comparisons_for_solver(comps)
    logging.info(f"[recalc] {len(filtered)}/{len(comps)} valid comparisons for solver")

    if not filtered:
        logging.warning("No valid comparisons – cannot solve.")
        return {}, {}

    models_for_solve = sorted(list(all_models | models_in_comparisons(filtered)))
    init_final = {m: DEFAULT_ELO for m in models_for_solve}

    # --- final μ  (bin 20) --------------------------------------------
    mu_map, _ = solve_with_trueskill(
        models_for_solve, filtered, init_final,
        debug=False, use_fixed_initial_ratings=True,
        bin_size=WIN_MARGIN_BIN_SIZE, return_sigma=True)

    # --- σ / CI  (bin 5) ----------------------------------------------
    _, sigma_map = solve_with_trueskill(
        models_for_solve, filtered, init_final,
        debug=False, use_fixed_initial_ratings=True,
        bin_size=WIN_MARGIN_BIN_SIZE_FOR_CI, return_sigma=True)

    return mu_map, sigma_map

# ─────────────────────────────────────────────────────────────────────
def _write_back_recalc(elo_path: Path, old_blob: Dict[str, Any],
                       mu: Dict[str, float], sigma: Dict[str, float]) -> bool:
    """Writes the recalculated ELO data back to the specified file."""

    ts_env_sigma = 350 / 3
    new_blob     = copy.deepcopy(old_blob) # Start with old metadata, comparisons etc.

    # Clear existing model entries before writing new ones
    models_to_clear = [k for k in new_blob if k != "__metadata__"]
    for k in models_to_clear:
        del new_blob[k]

    if not mu: # Handle case where solving failed or yielded no results
        logging.warning("[recalc] No ELO scores generated. Writing empty model data.")
        # Keep metadata, but no model entries
    else:
        # normalise μ and CI bounds exactly like benchmark
        norm_mu  = normalize_elo_scores(mu)

        raw_bounds = {}
        for m in mu:
            sig = sigma.get(m, ts_env_sigma)
            raw_bounds[f"{m}_low"] = mu[m] - 1.96 * sig
            raw_bounds[f"{m}_hi"]  = mu[m] + 1.96 * sig
        norm_bounds = normalize_elo_scores(raw_bounds)

        for m in mu:
            sig     = sigma.get(m, ts_env_sigma)
            ci_low  = raw_bounds.get(f"{m}_low", mu[m]) # Default to mu if bound missing
            ci_high = raw_bounds.get(f"{m}_hi", mu[m])
            norm_ci_low = norm_bounds.get(f"{m}_low", norm_mu[m])
            norm_ci_high = norm_bounds.get(f"{m}_hi", norm_mu[m])

            new_blob[m] = {
                "elo":          round(mu[m],            2),
                "elo_norm":     round(norm_mu[m],       2),
                "sigma":        round(sig,              2),
                "ci_low":       round(ci_low,           2),
                "ci_high":      round(ci_high,          2),
                "ci_low_norm":  round(norm_ci_low, 2),
                "ci_high_norm": round(norm_ci_high,  2),
            }

    new_blob.setdefault("__metadata__", {})
    new_blob["__metadata__"]["last_updated"] = datetime.now(
        timezone.utc).isoformat()

    # Use atomic save for single file write as well
    if _atomic_multi_save({str(elo_path): new_blob}):
        logging.info(f"[recalc] wrote refreshed ratings to {elo_path}")
        return True
    else:
        logging.error(f"[recalc] FAILED to save {elo_path}")
        return False

# ─────────────────────────────────────────────────────────────────────
def action_recalc(args, canonical_elo_data):
    """Full re-solve + overwrite canonical file so it matches pipeline."""
    logging.info("Starting full recalculation of canonical ELO...")
    canonical_elo_copy = copy.deepcopy(canonical_elo_data) # Work on a copy

    comps = canonical_elo_copy.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    if not comps:
        logging.error("No comparisons found in canonical data – aborting recalc.")
        return False

    # 1. refresh per-comparison stats (modifies copy in place)
    _refresh_comparison_fields(comps)

    # 2. union of models: those with comparisons + those already in blob
    models_comp = models_in_comparisons(comps)
    models_blob = {m for m in canonical_elo_copy if m != "__metadata__"}
    all_models  = models_comp.union(models_blob)
    if not all_models:
        logging.warning("[recalc] No models found in data after loading. Aborting.")
        return False
    logging.info(f"[recalc] solving for {len(all_models)} models")

    # 3. dual solve
    try:
        mu_map, sigma_map = _dual_solve(all_models, comps)
    except Exception as e:
        logging.error(f"[recalc] Solving failed: {e}", exc_info=True)
        return False

    # 4. write back (overwrites original file path using the copy's data)
    if not _write_back_recalc(Path(args.canonical_elo), canonical_elo_copy, mu_map, sigma_map):
        return False # Write back failed

    logging.info("Recalculation action completed successfully.")
    return True


# =========================================================================
# Main Execution
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Merge/Unmerge local runs/ELO with canonical files OR delete models from canonical files OR recalculate canonical ELO."
    )
    parser.add_argument(
        "--action",
        choices=["merge", "delete", "recalc", "unmerge"], # Added unmerge
        required=True, # Make action required
        help="Action to perform: 'merge' local->canonical, 'unmerge' canonical->local, 'delete' from canonical, 'recalc' canonical.",
    )
    parser.add_argument(
        "--local-runs",
        default=C.DEFAULT_LOCAL_RUNS_FILE,
        help="Path to the local runs JSON file (used for merge/unmerge).",
    )
    parser.add_argument(
        "--local-elo",
        default=C.DEFAULT_LOCAL_ELO_FILE,
        help="Path to the local ELO JSON file (used for merge/unmerge).",
    )
    parser.add_argument(
        "--canonical-runs",
        default=C.CANONICAL_LEADERBOARD_RUNS_FILE,
        help="Path to the canonical runs file (source/target).",
    )
    parser.add_argument(
        "--canonical-elo",
        default=C.CANONICAL_LEADERBOARD_ELO_FILE,
        help="Path to the canonical ELO file (source/target).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically confirm the selected action without prompting.",
    )
    parser.add_argument(
        "--verbosity",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity level.",
    )

    args = parser.parse_args()
    setup_merge_logging(args.verbosity)

    # --- Load Data ---
    logging.info(f"Loading canonical runs: {args.canonical_runs}")
    canonical_runs = load_json_file(args.canonical_runs)
    logging.info(f"Loading canonical ELO: {args.canonical_elo}")
    canonical_elo = load_json_file(args.canonical_elo)

    local_runs = {}
    local_elo = {}
    # Load local files if needed for merge or unmerge
    if args.action in ["merge", "unmerge"]:
        logging.info(f"Loading local runs: {args.local_runs}")
        local_runs = load_json_file(args.local_runs)
        logging.info(f"Loading local ELO: {args.local_elo}")
        local_elo = load_json_file(args.local_elo)
        if not isinstance(local_runs, dict) or not isinstance(local_elo, dict):
             logging.error("Failed to load local data files correctly for %s. Exiting.", args.action)
             sys.exit(1)

    if not isinstance(canonical_runs, dict) or not isinstance(canonical_elo, dict):
        logging.error("Failed to load canonical data files correctly. Exiting.")
        sys.exit(1)

    # Ensure metadata dicts exist
    canonical_runs.setdefault("__metadata__", {})
    canonical_elo.setdefault("__metadata__", {})
    if args.action in ["merge", "unmerge"]:
        local_runs.setdefault("__metadata__", {})
        local_elo.setdefault("__metadata__", {})

    files_to_save = None
    action_performed = False

    # --- Perform Action ---
    if args.action == "merge":
        candidates = find_merge_candidates(local_runs, local_elo, canonical_runs, canonical_elo)
        if not candidates:
            logging.info("No suitable candidates found for merging.")
            sys.exit(0)

        selected_candidates = select_models_from_list(candidates, "merge")
        if not selected_candidates:
            logging.info("No candidates selected for merging.")
            sys.exit(0)

        print("\n--- Summary of Merge ---")
        print(f"Models to merge: {', '.join(c['model_name'] for c in selected_candidates)}")
        print(f"Local Runs File:      {args.local_runs} (will be modified)")
        print(f"Local ELO File:       {args.local_elo} (will be modified)")
        print(f"Canonical Runs File:  {args.canonical_runs} (will be modified)")
        print(f"Canonical ELO File:   {args.canonical_elo} (will be modified)")
        print("------------------------")

        if not (args.yes or input("Proceed with merge? (y/n): ").strip().lower().startswith("y")):
            logging.info("Merge cancelled by user.")
            sys.exit(0)

        logging.info("Starting merge process…")
        local_runs_copy = copy.deepcopy(local_runs)
        local_elo_copy = copy.deepcopy(local_elo)
        canonical_runs_copy = copy.deepcopy(canonical_runs)
        canonical_elo_copy = copy.deepcopy(canonical_elo)

        if not merge_data(selected_candidates, local_runs_copy, local_elo_copy, canonical_runs_copy, canonical_elo_copy):
            logging.error("No data moved during merge; aborting.")
            sys.exit(1)

        logging.info("Recalculating canonical ELO after merge...")
        if not _recalculate_elo_ratings(canonical_elo_copy): # Use refactored function
            logging.error("Canonical ELO recalculation failed; no files will be written.")
            sys.exit(1)
        # Note: Local ELO doesn't need recalculation after merge as only removed items

        logging.info("Saving modified files…")
        files_to_save = {
            args.canonical_runs: canonical_runs_copy,
            args.canonical_elo : canonical_elo_copy,
            args.local_runs    : local_runs_copy,
            args.local_elo     : local_elo_copy,
        }
        action_performed = True

    elif args.action == "unmerge":
        candidates = find_unmerge_candidates(canonical_runs, canonical_elo)
        if not candidates:
            logging.info("No suitable candidates found for unmerging.")
            sys.exit(0)

        selected_candidates = select_models_from_list(candidates, "unmerge")
        if not selected_candidates:
            logging.info("No candidates selected for unmerging.")
            sys.exit(0)

        print("\n--- Summary of Unmerge ---")
        print(f"Models to unmerge: {', '.join(c['model_name'] for c in selected_candidates)}")
        print(f"Local Runs File:      {args.local_runs} (will be modified)")
        print(f"Local ELO File:       {args.local_elo} (will be modified)")
        print(f"Canonical Runs File:  {args.canonical_runs} (will be modified)")
        print(f"Canonical ELO File:   {args.canonical_elo} (will be modified)")
        print("--------------------------")

        if not (args.yes or input("Proceed with unmerge? (y/n): ").strip().lower().startswith("y")):
            logging.info("Unmerge cancelled by user.")
            sys.exit(0)

        logging.info("Starting unmerge process…")
        local_runs_copy = copy.deepcopy(local_runs)
        local_elo_copy = copy.deepcopy(local_elo)
        canonical_runs_copy = copy.deepcopy(canonical_runs)
        canonical_elo_copy = copy.deepcopy(canonical_elo)

        if not unmerge_data(selected_candidates, local_runs_copy, local_elo_copy, canonical_runs_copy, canonical_elo_copy):
            logging.error("No data moved during unmerge; aborting.")
            sys.exit(1)

        logging.info("Recalculating canonical ELO after unmerge...")
        recalc_canon_ok = _recalculate_elo_ratings(canonical_elo_copy)
        logging.info("Recalculating local ELO after unmerge...")
        recalc_local_ok = _recalculate_elo_ratings(local_elo_copy)

        if not (recalc_canon_ok and recalc_local_ok):
            logging.error("ELO recalculation failed for one or both files; no files will be written.")
            sys.exit(1)

        logging.info("Saving modified files…")
        files_to_save = {
            args.canonical_runs: canonical_runs_copy,
            args.canonical_elo : canonical_elo_copy,
            args.local_runs    : local_runs_copy,
            args.local_elo     : local_elo_copy,
        }
        action_performed = True

    elif args.action == "delete":
        candidates = find_delete_candidates(canonical_runs, canonical_elo)
        if not candidates:
            logging.info("No models found in canonical files to delete.")
            sys.exit(0)

        selected_models = select_models_from_list(candidates, "delete")
        if not selected_models:
            logging.info("No models selected for deletion.")
            sys.exit(0)

        print("\n--- Summary of Deletion ---")
        print(f"Models to delete: {', '.join(c['model_name'] for c in selected_models)}")
        print(f"Canonical Runs File:  {args.canonical_runs} (will be modified)")
        print(f"Canonical ELO File:   {args.canonical_elo} (will be modified)")
        print("---------------------------")

        if not (args.yes or input("Proceed with deletion? (y/n): ").strip().lower().startswith("y")):
            logging.info("Deletion cancelled by user.")
            sys.exit(0)

        logging.info("Starting deletion process…")
        canonical_runs_copy = copy.deepcopy(canonical_runs)
        canonical_elo_copy = copy.deepcopy(canonical_elo)

        if not delete_data(selected_models, canonical_runs_copy, canonical_elo_copy):
            logging.error("No data removed during deletion; aborting.")
            sys.exit(1)

        logging.info("Recalculating canonical ELO after deletion...")
        if not _recalculate_elo_ratings(canonical_elo_copy): # Use refactored function
            logging.error("Canonical ELO recalculation failed; no files will be written.")
            sys.exit(1)

        logging.info("Saving modified canonical files…")
        files_to_save = {
            args.canonical_runs: canonical_runs_copy,
            args.canonical_elo : canonical_elo_copy,
        }
        action_performed = True

    elif args.action == "recalc":
        # Recalc action handles its own saving internally via _write_back_recalc
        if not action_recalc(args, canonical_elo):
             logging.error("Recalculation action failed.")
             sys.exit(1)
        # No need to set files_to_save or action_performed here
        logging.info("Recalculation action completed.")
        sys.exit(0) # Exit after recalc action

    else:
        # Should be caught by argparse choices
        logging.error(f"Invalid action specified: {args.action}")
        sys.exit(1)

    # --- Perform Save (for merge, unmerge, delete) ---
    if action_performed and files_to_save:
        if _atomic_multi_save(files_to_save):
            logging.info(f"{args.action.capitalize()} process completed successfully.")
        else:
            logging.error(f"{args.action.capitalize()} aborted – atomic save failed, no files should have been overwritten.")
            sys.exit(1)
    elif action_performed and not files_to_save:
         logging.error("Action was marked as performed, but no files were set to be saved. This indicates an internal logic error.")
         sys.exit(1)
    # else: action was not performed (e.g., user cancelled) or was 'recalc' which saves itself.


if __name__ == "__main__":
    main()
