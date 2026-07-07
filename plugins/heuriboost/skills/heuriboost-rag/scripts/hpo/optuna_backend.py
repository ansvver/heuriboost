#!/usr/bin/env python3
"""Optuna backend for the HPOEngine (spec §12.5).

Searches XGBoost params under a fixed budget on fixed train+valid snapshots.
Deterministic via `TPESampler(seed=...)` + `nthread=1` per trial. Case-blind
and test-blind (signatures take only snapshots). Trial failures are recorded,
not raised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import ndcg_at_k
from hpo.engine import Budget, TrialResult
from ranking_snapshot import Snapshot

# Fixed training constants for every HPO trial.
OBJECTIVE = "rank:ndcg"
EVAL_METRIC = "ndcg@10"
NUM_BOOST_ROUND = 200
EARLY_STOPPING_ROUNDS = 20
FIXED_SEED = 42
NTHREAD = 1  # full determinism (xgboost multi-thread histogram building is non-deterministic)
SKILL_DIR = Path(__file__).resolve().parents[2]


def _require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna is required for HPO. Install with: python -m pip install -r "
            f"{SKILL_DIR / 'requirements-build.txt'}"
        ) from exc
    return optuna


def _ndcg10_from_scores(scores, raw_labels, groups) -> float:
    """Compute nDCG@10 from raw prediction scores + ORIGINAL labels + group sizes.

    Uses the SAME `ndcg_at_k` formula as `evaluate_ranked_frame` (which clamps
    raw -1 to 0 via `max(label, 0)`), so HPO scores are directly comparable to
    the shipped baseline (0.853) and the post-hoc test eval. NOT the mapped
    0..4 labels (which would use a different gain scale).
    """
    import pandas as pd

    frame = pd.DataFrame({"score": list(scores), "label": list(raw_labels), "group": 0})
    gids = []
    for gi, size in enumerate(groups):
        gids.extend([gi] * size)
    frame["group"] = gids
    total = 0.0
    for _, g in frame.groupby("group", sort=False):
        ordered = g.sort_values("score", ascending=False, kind="stable")["label"].tolist()
        total += ndcg_at_k(ordered, 10)
    return total / max(len(groups), 1)


class OptunaBackend:
    """TPESampler-based Optuna backend with a narrowed V0 search space."""

    def __init__(self, search_space: dict[str, tuple] | None = None) -> None:
        # Narrowed around the shipped baseline (150 train / 40 validation queries
        # cannot support wide bounds). Confirmed during grilling.
        self._search_space = search_space or {
            "max_depth": (3, 6),
            "min_child_weight": (0.05, 3.0),
            "eta": (0.05, 0.2),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "gamma": (0.0, 3.0),
            "reg_lambda": (0.1, 10.0),
        }

    def run(
        self,
        train_snapshot: Snapshot,
        valid_snapshot: Snapshot,
        budget: Budget,
        seed: int,
        feature_set_name: str,
        feature_set_version: int,
        task_profile: str,
    ) -> TrialResult:
        import xgboost as xgb

        optuna = _require_optuna()

        dtrain = xgb.DMatrix(
            train_snapshot.X, label=train_snapshot.y
        )
        dtrain.set_group(train_snapshot.groups)
        dvalid = xgb.DMatrix(
            valid_snapshot.X, label=valid_snapshot.y
        )
        dvalid.set_group(valid_snapshot.groups)

        feature_names = list(train_snapshot.X.columns)

        def _suggest(trial) -> dict:
            sp = self._search_space
            return {
                "max_depth": trial.suggest_int("max_depth", *sp["max_depth"]),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", *sp["min_child_weight"], log=True
                ),
                "eta": trial.suggest_float("eta", *sp["eta"], log=True),
                "subsample": trial.suggest_float("subsample", *sp["subsample"]),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *sp["colsample_bytree"]
                ),
                "gamma": trial.suggest_float("gamma", *sp["gamma"]),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *sp["reg_lambda"], log=True
                ),
            }

        def _objective(trial) -> float:
            params = _suggest(trial)
            params.update(
                {
                    "objective": OBJECTIVE,
                    "eval_metric": EVAL_METRIC,
                    "seed": FIXED_SEED,
                    "nthread": NTHREAD,
                }
            )
            try:
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=NUM_BOOST_ROUND,
                    evals=[(dvalid, "validation")],
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )
                # Use the SAME ndcg@10 formula as the shipped baseline (not
                # xgboost's internal best_score, which uses a different ndcg
                # computation and would make HPO scores incomparable to 0.853
                # and to the post-hoc test eval). Predict with iteration_range
                # = (0, best_iteration+1) so the score reflects the
                # best-iteration model (the one a consumer reproduces), not the
                # full early-stopping-window model.
                best_iter = int(model.best_iteration)
                preds = model.predict(dvalid, iteration_range=(0, best_iter + 1))
                score = _ndcg10_from_scores(preds, valid_snapshot.raw_labels, valid_snapshot.groups)
                # stash for history collection
                trial.set_user_attr("score", score)
                trial.set_user_attr("best_iteration", int(model.best_iteration))
                trial.set_user_attr("state", "complete")
                trial.set_user_attr("failure_reason", None)
            except Exception as exc:
                trial.set_user_attr("score", float("nan"))
                trial.set_user_attr("best_iteration", -1)
                trial.set_user_attr("state", "failed")
                trial.set_user_attr("failure_reason", str(exc))
                return float("nan")  # Optuna prunes NaN objectives
            return score

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        try:
            study.optimize(
                _objective,
                n_trials=budget.n_trials,
                timeout=budget.timeout_sec,
                catch=(Exception,),
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            pass  # return best-so-far

        trials_history = []
        for tr in study.trials:
            if tr.state.name == "COMPLETE" or tr.user_attrs.get("state") == "failed":
                trials_history.append(
                    {
                        "number": tr.number,
                        "params": dict(tr.params),
                        "score": float(tr.user_attrs.get("score", float("nan"))),
                        "state": tr.user_attrs.get("state", tr.state.name.lower()),
                        "failure_reason": tr.user_attrs.get("failure_reason"),
                    }
                )

        # Best trial = highest complete score.
        complete = [t for t in trials_history if t["state"] == "complete"]
        if not complete:
            raise SystemExit(
                "HPO failed: no complete trials (all failed). Check trial failure_reason."
            )
        best = max(complete, key=lambda t: t["score"])
        # Re-run the best params once to capture best_iteration deterministically.
        # xgboost `best_iteration` is 0-indexed; reproducing the best model
        # requires num_boost_round = best_iteration + 1 (train rounds 0..best_iteration).
        best_params = dict(best["params"])
        best_params.update(
            {
                "objective": OBJECTIVE,
                "eval_metric": EVAL_METRIC,
                "seed": FIXED_SEED,
                "nthread": NTHREAD,
            }
        )
        best_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        best_iteration = int(best_model.best_iteration)

        return TrialResult(
            best_params=dict(best["params"]),
            best_score=float(best["score"]),
            best_iteration=best_iteration,
            trials=trials_history,
            feature_set_name=feature_set_name,
            feature_set_version=feature_set_version,
            objective=OBJECTIVE,
            eval_metric=EVAL_METRIC,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            num_boost_round=NUM_BOOST_ROUND,
            seed=seed,
            n_trials=budget.n_trials,
            timeout_sec=budget.timeout_sec,
        )
