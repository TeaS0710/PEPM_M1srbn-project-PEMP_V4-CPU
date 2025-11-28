#!/usr/bin/env python
"""
post_analysis_generic.py

Post-analyse générique pour les expériences lancées via superior_orchestrator.

- Lit superior/<exp_id>/metrics_global.tsv
- Convertit les colonnes numériques quand c'est possible
- Trace des courbes pour une ou plusieurs métriques en fonction d'un axe x,
  groupées par un ou plusieurs champs (ex: family, dataset_id, axis_dataset).

Ce script NE dépend d'aucune config spécifique (idéologie, balance, etc.).
Il se contente d'exploiter les colonnes disponibles dans metrics_global.tsv.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


META_COLUMNS_DEFAULT = {
    "run_id",
    "exp_id",
    "profile",
    "stage",
    "status",
    "return_code",
    "family",
    "model_id",
    "corpus_id",
    "dataset_id",
    "view",
    "train_prop",
    "TRAIN_PROP",
    "axis_values_json",
    "make_vars_json",
    "overrides_json",
    "metrics_path",
    "log_path",
    "started_at",
    "finished_at",
    "duration_s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generic post-analysis over superior metrics_global.tsv"
    )
    parser.add_argument(
        "--exp-id",
        required=True,
        help="Identifiant d'expérience (dossier sous superior/)",
    )
    parser.add_argument(
        "--metrics-path",
        help="Chemin vers metrics_global.tsv (sinon: superior/<exp-id>/metrics_global.tsv)",
    )
    parser.add_argument(
        "--outdir",
        help="Dossier de sortie pour les graphes (défaut: superior/<exp-id>/analysis)",
    )
    parser.add_argument(
        "--x-axis",
        help="Colonne à utiliser comme axe X (défaut: TRAIN_PROP si présent, sinon auto)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help=(
            "Liste des colonnes métriques à tracer. "
            "Défaut: détecte automatiquement accuracy / macro_f1 / f1 / precision / recall si présentes."
        ),
    )
    parser.add_argument(
        "--group-by",
        nargs="*",
        help=(
            "Colonnes de regroupement (par ex: family dataset_id axis_dataset). "
            "Défaut: ['family','dataset_id'] si dispo, sinon aucun groupement."
        ),
    )
    return parser.parse_args()


def load_metrics_table(path: Path) -> List[Dict[str, Any]]:
    """Charge metrics_global.tsv et tente de caster les valeurs numériques."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows_raw = list(reader)

    rows: List[Dict[str, Any]] = []
    for row in rows_raw:
        parsed_row: Dict[str, Any] = {}
        for key, value in row.items():
            if value in (None, ""):
                parsed_row[key] = ""
                continue
            # on tente float -> sinon string
            try:
                parsed_row[key] = float(value)
            except ValueError:
                parsed_row[key] = value
        rows.append(parsed_row)
    return rows


def infer_x_axis(rows: List[Dict[str, Any]], preferred: str | None = None) -> str | None:
    if not rows:
        return None
    columns = rows[0].keys()

    # 1) Si l'utilisateur a donné un x-axis et qu'il existe
    if preferred and preferred in columns:
        return preferred

    # 2) TRAIN_PROP (float) est souvent ce qu'on veut
    if "TRAIN_PROP" in columns:
        return "TRAIN_PROP"
    if "train_prop" in columns:
        return "train_prop"

    # 3) Sinon: premier champ numérique non meta
    for col in columns:
        if col in META_COLUMNS_DEFAULT:
            continue
        # On teste sur le premier row
        val = rows[0].get(col)
        if isinstance(val, (int, float, float)):
            return col

    return None


def infer_metric_columns(rows: List[Dict[str, Any]], x_axis: str | None) -> List[str]:
    if not rows:
        return []

    columns = rows[0].keys()
    candidates_named = [
        "accuracy",
        "macro_f1",
        "micro_f1",
        "f1",
        "precision",
        "recall",
    ]

    # 1) Priorité aux noms de métriques connus
    metrics = [c for c in candidates_named if c in columns]
    if metrics:
        # On enlève l'axe X s'il traîne là
        metrics = [m for m in metrics if m != x_axis]
        if metrics:
            return metrics

    # 2) Sinon: tous les champs numériques hors meta et hors x_axis
    numeric_metrics: List[str] = []
    for col in columns:
        if col == x_axis:
            continue
        if col in META_COLUMNS_DEFAULT:
            continue
        val = rows[0].get(col)
        if isinstance(val, (int, float)):
            numeric_metrics.append(col)
    return numeric_metrics


def infer_group_by(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    cols = set(rows[0].keys())

    # Heuristique simple : family / dataset_id si présents
    group_pref = ["family", "dataset_id", "view"]
    found = [c for c in group_pref if c in cols]
    # Ajout possible d'un axe dataset (axis_dataset) si dispo
    if "axis_dataset" in cols and "axis_dataset" not in found:
        found.append("axis_dataset")
    return found


def group_rows(
    rows: List[Dict[str, Any]],
    group_by: List[str],
) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    if not group_by:
        # un seul groupe global
        grouped[tuple()] = rows
        return grouped

    for row in rows:
        key = tuple(row.get(col, "") for col in group_by)
        grouped.setdefault(key, []).append(row)
    return grouped


def plot_curves(
    rows: List[Dict[str, Any]],
    x_axis: str,
    metrics: List[str],
    group_by: List[str],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        grouped = group_rows(rows, group_by)
        has_data = False

        plt.figure()
        for group_key, rs in grouped.items():
            # collecter les points (x,y) numériques
            points: List[Tuple[float, float]] = []
            for r in rs:
                x_val = r.get(x_axis)
                y_val = r.get(metric)
                if not isinstance(x_val, (int, float)) or not isinstance(y_val, (int, float)):
                    continue
                points.append((x_val, y_val))

            if not points:
                continue

            has_data = True
            points_sorted = sorted(points, key=lambda p: p[0])
            xs, ys = zip(*points_sorted)
            label = ", ".join(str(v) for v in group_key if v not in ("", None))
            plt.plot(xs, ys, marker="o", label=label or None)

        if not has_data:
            print(f"[post_analysis] Aucun data pour metric '{metric}' vs '{x_axis}', skip.")
            plt.close()
            continue

        plt.xlabel(x_axis)
        plt.ylabel(metric)
        title = f"{metric} vs {x_axis}"
        if group_by:
            title += " (" + ", ".join(group_by) + ")"
        plt.title(title)
        if group_by:
            plt.legend()
        plt.grid(True, linestyle=":", alpha=0.4)
        plt.tight_layout()

        out_path = outdir / f"{metric}__vs__{x_axis}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"[post_analysis] Wrote {out_path}")


def main() -> None:
    args = parse_args()

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
    else:
        metrics_path = Path("superior") / args.exp_id / "metrics_global.tsv"

    if not metrics_path.exists():
        raise SystemExit(f"[post_analysis] metrics_global.tsv not found at {metrics_path}")

    rows = load_metrics_table(metrics_path)
    if not rows:
        raise SystemExit("[post_analysis] metrics_global.tsv is empty")

    # Déterminer l'axe X
    x_axis = infer_x_axis(rows, preferred=args.x_axis)
    if not x_axis:
        print("[post_analysis] Impossible de déterminer un x_axis par défaut.")
        print("Colonnes disponibles :", ", ".join(rows[0].keys()))
        return
    print(f"[post_analysis] Using x_axis = {x_axis}")

    # Déterminer les métriques à tracer
    if args.metrics:
        metrics = args.metrics
    else:
        metrics = infer_metric_columns(rows, x_axis=x_axis)
    if not metrics:
        print("[post_analysis] Aucune métrique numérique détectée à tracer.")
        print("Colonnes disponibles :", ", ".join(rows[0].keys()))
        return
    print(f"[post_analysis] Metrics = {', '.join(metrics)}")

    # Déterminer les colonnes de group_by
    if args.group_by is not None:
        group_by = args.group_by
    else:
        group_by = infer_group_by(rows)
    print(
        "[post_analysis] Group by = "
        + (", ".join(group_by) if group_by else "<no grouping>")
    )

    # Dossier de sortie
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = metrics_path.parent / "analysis"

    plot_curves(rows, x_axis=x_axis, metrics=metrics, group_by=group_by, outdir=outdir)


if __name__ == "__main__":
    main()
