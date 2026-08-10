from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from build_analytics import BUILDERS, _selected_families  # noqa: E402
from core.analytics_manifest import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    RUNTIME_ARTIFACTS,
    build_manifest,
    get_refresh_status,
    inspect_runtime_artifacts,
    validate_runtime_artifacts,
    write_manifest,
)

TMP_DIR_NAME = ".analytics_build_tmp"
BACKUP_DIR_NAME = ".analytics_backups"
MANIFEST_REL_PATH = "data/analytics/manifest.json"
PROMOTED_REL_PATHS = tuple(RUNTIME_ARTIFACTS) + (MANIFEST_REL_PATH,)

SOURCE_FAMILY_MAP = {
    "data/xlsx/tratado_safeguardoffshore.xlsx": "sphera",
    "data/xlsx/dicionarioweaksignals.xlsx": "ws",
    "data/xlsx/precursores_expandido.xlsx": "precursors",
    "data/xlsx/taxonomiacp_por.xlsx": "cp",
}


def _print_json(title: str, payload: dict) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _all_changed_sources(status: dict) -> list[str]:
    diff = status.get("diff", {}) or {}
    paths = []
    for key in ("added", "changed", "removed"):
        paths.extend(diff.get(key, []) or [])
    return sorted(set(paths))


def _has_xlsx_change(status: dict) -> bool:
    return any(path.casefold().startswith("data/xlsx/") for path in _all_changed_sources(status))


def _select_auto_families(status: dict) -> list[str]:
    if status.get("missing_artifacts"):
        return list(BUILDERS)

    families = set()
    unknown_xlsx = []
    for path in _all_changed_sources(status):
        folded = path.casefold()
        if not folded.startswith("data/xlsx/"):
            continue
        family = SOURCE_FAMILY_MAP.get(folded)
        if family:
            families.add(family)
        else:
            unknown_xlsx.append(path)

    if unknown_xlsx:
        print("Planilha sem mapeamento especifico detectada; reconstruindo todas as familias:")
        for path in unknown_xlsx:
            print(f"  - {path}")
        return list(BUILDERS)

    ordered = [family for family in BUILDERS if family in families]
    return ordered or list(BUILDERS)


def _safe_rmtree(path: Path, repo_root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    root = repo_root.resolve()
    if not resolved.is_relative_to(root):
        raise RuntimeError(f"Recusa em remover caminho fora do repositorio: {resolved}")
    if path.name not in {TMP_DIR_NAME}:
        raise RuntimeError(f"Recusa em remover pasta inesperada: {resolved}")
    shutil.rmtree(resolved)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        dst.mkdir(parents=True, exist_ok=True)
        return
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _prepare_staging_root(repo_root: Path, staging_root: Path) -> None:
    _safe_rmtree(staging_root, repo_root)
    (staging_root / "data").mkdir(parents=True, exist_ok=True)

    _copy_tree_if_exists(repo_root / "data" / "xlsx", staging_root / "data" / "xlsx")
    _copy_tree_if_exists(repo_root / "data" / "docs", staging_root / "data" / "docs")
    (staging_root / "data" / "analytics").mkdir(parents=True, exist_ok=True)

    # Copia os artefatos atuais para permitir builds parciais com validacao completa.
    for rel_path in PROMOTED_REL_PATHS:
        src = repo_root / rel_path
        if src.exists():
            _copy_file(src, staging_root / rel_path)


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst.with_name(f"{dst.name}.tmp")
    shutil.copy2(src, tmp_dst)
    os.replace(tmp_dst, dst)


def _backup_current_artifacts(repo_root: Path) -> tuple[Path, dict[str, bool]]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = repo_root / BACKUP_DIR_NAME / stamp
    backup_dir.mkdir(parents=True, exist_ok=True)

    existed: dict[str, bool] = {}
    for rel_path in PROMOTED_REL_PATHS:
        src = repo_root / rel_path
        existed[rel_path] = src.exists()
        if src.exists():
            _copy_file(src, backup_dir / rel_path)

    return backup_dir, existed


def _restore_backup(repo_root: Path, backup_dir: Path, existed: dict[str, bool]) -> None:
    for rel_path, was_present in existed.items():
        dst = repo_root / rel_path
        backup = backup_dir / rel_path
        if was_present and backup.exists():
            _copy_atomic(backup, dst)
        elif not was_present and dst.exists():
            dst.unlink()


def _promote_staging(staging_root: Path, repo_root: Path) -> tuple[Path, dict[str, bool]]:
    backup_dir, existed = _backup_current_artifacts(repo_root)
    try:
        for rel_path in PROMOTED_REL_PATHS:
            src = staging_root / rel_path
            if not src.exists():
                raise RuntimeError(f"Artefato gerado ausente no staging: {rel_path}")
            _copy_atomic(src, repo_root / rel_path)
    except Exception:
        print("\nFalha durante a substituicao dos artefatos. Restaurando backup...")
        _restore_backup(repo_root, backup_dir, existed)
        raise
    return backup_dir, existed


def _validate_or_raise(repo_root: Path, title: str) -> dict:
    artifacts = inspect_runtime_artifacts(repo_root)
    validation = validate_runtime_artifacts(artifacts)
    _print_json(title, validation)
    if validation["errors"]:
        raise RuntimeError("Validacao falhou; os artefatos atuais nao serao substituidos.")
    return validation


def _sync_manifest_only(repo_root: Path, model_name: str) -> int:
    validation = _validate_or_raise(repo_root, "Validacao dos artefatos atuais")
    artifacts = inspect_runtime_artifacts(repo_root)
    manifest = build_manifest(
        repo_root,
        artifacts=artifacts,
        generated_by="tools/update_analytics_local.py",
        mode="docs-only-manifest-sync",
        embedding_model=model_name,
    )
    manifest["validation"] = validation
    path = write_manifest(repo_root, manifest)
    print(f"\nManifesto atualizado sem regerar embeddings: {path}")
    _print_json("Status final", get_refresh_status(repo_root))
    return 0


def _build_in_staging(
    repo_root: Path,
    staging_root: Path,
    model_name: str,
    families: list[str],
) -> dict[str, dict[str, int]]:
    _prepare_staging_root(repo_root, staging_root)

    summaries = {}
    for family in families:
        print(f"\nGerando familia em staging: {family}")
        summaries[family] = BUILDERS[family](staging_root, model_name)

    artifacts = inspect_runtime_artifacts(staging_root)
    validation = validate_runtime_artifacts(artifacts)
    _print_json("Validacao no staging", validation)
    if validation["errors"]:
        raise RuntimeError("Staging invalido; data/analytics real nao foi alterado.")

    manifest = build_manifest(
        staging_root,
        artifacts=artifacts,
        generated_by="tools/update_analytics_local.py",
        mode="local-safe-build",
        embedding_model=model_name,
    )
    manifest["build_summary"] = summaries
    manifest["validation"] = validation
    write_manifest(staging_root, manifest)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Atualiza localmente os artefatos de analytics do Safety Chat com staging e backup."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--model", default=os.getenv("ST_MODEL_NAME", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument(
        "--families",
        default="auto",
        help="auto, all ou lista separada por virgula: sphera,ws,precursors,cp",
    )
    parser.add_argument("--force", action="store_true", help="Regera mesmo quando nao ha mudancas detectadas.")
    parser.add_argument("--dry-run", action="store_true", help="Gera e valida no staging, mas nao substitui arquivos.")
    parser.add_argument("--keep-temp", action="store_true", help="Mantem .analytics_build_tmp para inspecao.")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    staging_root = repo_root / TMP_DIR_NAME

    status = get_refresh_status(repo_root)
    _print_json("Status inicial", status)

    if not args.force and not status.get("stale"):
        print("\nNenhuma alteracao detectada. Nada a atualizar.")
        return 0

    if not args.force and status.get("stale") and not _has_xlsx_change(status) and not status.get("missing_artifacts"):
        print(
            "\nApenas fontes em data/docs mudaram. O runtime atual nao gera embeddings "
            "dos documentos; validando analytics e sincronizando o manifesto."
        )
        return _sync_manifest_only(repo_root, args.model)

    families = _select_auto_families(status) if args.families == "auto" else _selected_families(args.families)
    print("\nFamilias selecionadas: " + ", ".join(families))

    try:
        summaries = _build_in_staging(repo_root, staging_root, args.model, families)
        _print_json("Resumo da geracao", summaries)

        if args.dry_run:
            print("\nDry-run concluido. Nenhum arquivo em data/analytics foi substituido.")
            return 0

        backup_dir, existed = _promote_staging(staging_root, repo_root)
        print(f"\nBackup dos artefatos anteriores: {backup_dir}")

        try:
            real_validation = _validate_or_raise(repo_root, "Validacao final em data/analytics")
            final_status = get_refresh_status(repo_root)
            _print_json("Status final", final_status)
            if final_status.get("stale"):
                raise RuntimeError("Manifesto final ainda indica fontes desatualizadas.")
        except Exception:
            print("\nValidacao final falhou. Restaurando backup dos artefatos anteriores...")
            _restore_backup(repo_root, backup_dir, existed)
            raise

        print("\nAtualizacao concluida com sucesso.")
        if real_validation["warnings"]:
            print("Revise os warnings antes de publicar no GitHub.")
        return 0
    finally:
        if not args.keep_temp:
            _safe_rmtree(staging_root, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
