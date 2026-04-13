import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional


# Vetor de pistas (edite aqui quando quiser)
TRACKS_TO_RUN = [
    "Bahrain Grand Prix",
    "Hungarian Grand Prix",
    "Italian Grand Prix",
    "Saudi Arabian Grand Prix",
    "United States Grand Prix",
]

# Modelos que serao executados por padrao (chaves de AVAILABLE_MODELS)
DEFAULT_MODELS_TO_RUN = [
    "model_lr_cv",
]

AVAILABLE_MODELS: Dict[str, str] = {
    "model_lr_cv": "model_lr_cv.py",
}


def safe_name(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def model_family_and_approach(model_key: str) -> tuple[str, str]:
    if model_key.startswith("model_lr_"):
        family = "linear_regression"
    elif model_key.startswith("model_xgb_"):
        family = "xgboost"
    else:
        family = "other"

    approach = model_key.split("_")[-1]
    return family, approach


def parse_metric(output: str, metric: str, kind: str) -> Optional[float]:
    """
    kind: 'mean' or 'holdout'
    """
    if kind == "mean":
        # Ex: RMSE Medio: 1.2345 / RMSE Medio: / RMSE Medio:
        # Ex: RMSE Medio / RMSE Medio com acento
        patterns = [
            rf"{metric}\s+M[eEéÉ]dio:\s*([-+]?\d*\.?\d+)",
            rf"{metric}\s+M[eEéÉ]dia:\s*([-+]?\d*\.?\d+)",
        ]
    else:
        patterns = [rf"Holdout\s+{metric}:\s*([-+]?\d*\.?\d+)"]

    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None

    return None


def normalize_model_key(model_name: str) -> str:
    key = model_name.strip().lower()
    if key.endswith(".py"):
        key = key[:-3]
    return key


def run_one_model(
    source_dir: str,
    output_root_dir: str,
    run_id: str,
    track_name: str,
    model_key: str,
    timeout_seconds: Optional[int],
) -> Dict[str, object]:
    script_name = AVAILABLE_MODELS[model_key]
    script_path = os.path.join(source_dir, script_name)

    env = os.environ.copy()
    env["TARGET_GP_NAME"] = track_name

    start = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, script_path],
            cwd=source_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        status = "success" if completed.returncode == 0 else "failed"
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        output = f"{stdout}\n{stderr}".strip()
        error_message = "" if status == "success" else (stderr.strip() or "Falha sem mensagem em stderr")
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        output = f"{(exc.stdout or "")}\n{(exc.stderr or "")}".strip()
        error_message = f"Timeout apos {timeout_seconds} segundos"
        return_code = -1

    elapsed = round(time.time() - start, 2)

    family, approach = model_family_and_approach(model_key)
    output_logs_dir = os.path.join(output_root_dir, family, approach, "runs", run_id, "logs")
    os.makedirs(output_logs_dir, exist_ok=True)

    log_filename = f"{safe_name(track_name)}__{model_key}.log"
    log_path = os.path.join(output_logs_dir, log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)

    record: Dict[str, object] = {
        "track": track_name,
        "family": family,
        "approach": approach,
        "model": model_key,
        "script": script_name,
        "status": status,
        "return_code": return_code,
        "duration_seconds": elapsed,
        "rmse_mean": parse_metric(output, "RMSE", "mean"),
        "mae_mean": parse_metric(output, "MAE", "mean"),
        "r2_mean": parse_metric(output, "R2", "mean"),
        "rmse_holdout": parse_metric(output, "RMSE", "holdout"),
        "mae_holdout": parse_metric(output, "MAE", "holdout"),
        "r2_holdout": parse_metric(output, "R2", "holdout"),
        "error_message": error_message,
        "log_path": log_path,
    }

    return record


def write_csv(csv_path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = [
        "track",
        "family",
        "approach",
        "model",
        "script",
        "status",
        "return_code",
        "duration_seconds",
        "rmse_mean",
        "mae_mean",
        "r2_mean",
        "rmse_holdout",
        "mae_holdout",
        "r2_holdout",
        "error_message",
        "log_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa varios scripts de modelo para varias pistas e salva um consolidado de resultados."
        )
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        help="Lista de pistas. Se omitido, usa TRACKS_TO_RUN do script.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Lista de modelos (chaves de AVAILABLE_MODELS). "
            "Aceita com ou sem .py, ex: model_xgb_cv ou model_xgb_cv.py"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout em segundos para cada execucao individual. Padrao: sem timeout.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Para a execucao no primeiro erro.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.dirname(source_dir)
    results_root = os.path.join(scripts_dir, "Results")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(results_root, "runs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    tracks = args.tracks if args.tracks else TRACKS_TO_RUN

    if args.models:
        model_keys = [normalize_model_key(m) for m in args.models]
    else:
        model_keys = DEFAULT_MODELS_TO_RUN

    invalid_models = [m for m in model_keys if m not in AVAILABLE_MODELS]
    if invalid_models:
        valid_keys = ", ".join(sorted(AVAILABLE_MODELS.keys()))
        raise ValueError(
            f"Modelos invalidos: {invalid_models}. Modelos validos: {valid_keys}"
        )

    total = len(tracks) * len(model_keys)
    count = 0
    rows: List[Dict[str, object]] = []

    print("=== EXECUCAO EM LOTE DE MODELOS ===")
    print(f"Pistas: {tracks}")
    print(f"Modelos: {model_keys}")
    print(f"Total de execucoes: {total}")
    print(f"Pasta de saida: {output_dir}\n")

    for track in tracks:
        for model_key in model_keys:
            count += 1
            print(f"[{count}/{total}] Rodando {model_key} em '{track}'...")
            result = run_one_model(
                source_dir=source_dir,
                output_root_dir=results_root,
                run_id=run_id,
                track_name=track,
                model_key=model_key,
                timeout_seconds=args.timeout,
            )
            rows.append(result)

            print(
                f" -> status={result['status']} | duracao={result['duration_seconds']}s "
                f"| rmse_mean={result['rmse_mean']} | rmse_holdout={result['rmse_holdout']}"
            )

            if result["status"] != "success" and args.stop_on_error:
                print("Parando por causa de erro (--stop-on-error).")
                break

        if args.stop_on_error and rows and rows[-1]["status"] != "success":
            break

    summary_json_path = os.path.join(output_dir, "summary.json")
    summary_csv_path = os.path.join(output_dir, "summary.csv")

    payload = {
        "created_at": datetime.now().isoformat(),
        "tracks": tracks,
        "models": model_keys,
        "total_runs": len(rows),
        "success_runs": sum(1 for r in rows if r["status"] == "success"),
        "failed_runs": sum(1 for r in rows if r["status"] in {"failed", "timeout"}),
        "results": rows,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_csv(summary_csv_path, rows)

    print("\n=== FINALIZADO ===")
    print(f"JSON: {summary_json_path}")
    print(f"CSV:  {summary_csv_path}")
    print(f"Logs por modelo em: {results_root}/<family>/<approach>/runs/{run_id}/logs")


if __name__ == "__main__":
    main()
