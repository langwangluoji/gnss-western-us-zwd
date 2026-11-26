"""Build a source-linked inventory without loading observations or model arrays.

Only explicitly selected, small Markdown/JSON metadata files are opened.  The
catalog is a navigation aid: a reported status is not an independent scientific
verification, and all original reports, locks and numerical values stay intact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

OLD_ENGINE = Path("research_pipeline/studies/us_dense_zwd_20261105_v1/03_code/gnss_field_server")
MAX_METADATA_BYTES = 512 * 1024
MAX_DEPTH = 5
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", "cache", "caches", "data", "datasets",
    "labels", "observations", "checkpoints", "weights", "figures", "raw", "arrays",
}
JSON_NAMES = {
    "status.json", "progress.json", "metrics.json", "results.json", "summary.json",
    "comparison.json", "completion_snapshot.json", "delivery_verification.json",
    "training_verification.json", "review_receipt.json", "point_results.json",
    "probability_results.json", "verification.json", "experiment.json", "protocol.json",
    "config.json", "manifest.json",
}
TEXT_STEMS = {
    "readme", "report", "results", "research", "plan", "protocol", "progress",
    "review", "verdict", "handoff", "architecture", "model_plan", "experiment_lock",
}
FIELD_KEYS = {
    "status", "stage", "state", "completion_status", "objective", "purpose", "goal",
    "description", "experiment_role", "evidence_role", "evaluation_scope", "scope",
    "data_role", "evaluation_year", "train_years", "validation_years", "test_years",
    "development_exposed", "confirmation_labels_read", "heldout_labels_read",
}
STATUS_KEYS = {"status", "stage", "state", "completion_status"}


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def _link(path: Path, document: Path, label: str | None = None) -> str:
    target = quote(_relative(path, document.parent), safe="/._-")
    return f"[{label or path.name}]({target})"


def _kind(path: Path) -> str | None:
    name, stem = path.name.lower(), path.stem.lower()
    if path.suffix.lower() == ".md" and (stem in TEXT_STEMS or "report" in stem):
        return "report" if stem in {"report", "results", "review", "verdict"} or "report" in stem else "documentation"
    if path.suffix.lower() == ".json":
        if name.endswith("lock.json") or name.endswith(".lock.json") or "protocol" in name:
            return "protocol"
        if name in JSON_NAMES:
            return "status" if name in {"status.json", "progress.json", "completion_snapshot.json"} else "metadata"
    return None


def _is_link(path: Path) -> bool:
    # os.walk(followlinks=False) alone does not exclude Windows junctions.
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def _artifacts(directory: Path):
    """Enumerate metadata names only; never descend into data/weight directories."""
    for current, dirs, files in os.walk(directory, followlinks=False):
        here = Path(current)
        depth = len(here.relative_to(directory).parts)
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS and not _is_link(here / d)) if depth < MAX_DEPTH else []
        for name in sorted(files):
            path = here / name
            kind = _kind(path)
            if kind and not _is_link(path):
                yield path, kind


def _fields(value, prefix="", depth=0):
    """Extract semantic scalar fields, not station, label or score arrays."""
    if not isinstance(value, dict) or depth > 4:
        return
    for key, item in value.items():
        key = str(key)
        qualified = f"{prefix}.{key}" if prefix else key
        if key.lower() in FIELD_KEYS and isinstance(item, (str, int, float, bool)):
            yield {"key": qualified, "value": str(item)[:700]}
        elif isinstance(item, dict) and key.lower() not in {
            "stations", "labels", "predictions", "targets", "station_metrics", "per_station",
        }:
            yield from _fields(item, qualified, depth + 1)


def _read_artifact(path: Path, kind: str, root: Path):
    record = {"path": _relative(path, root), "kind": kind, "bytes": path.stat().st_size}
    if record["bytes"] > MAX_METADATA_BYTES:
        record["read_status"] = "linked_only_size_limit"
        return record, [], []
    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8-sig")
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        record["read_status"] = "read"
        if path.suffix.lower() == ".json":
            fields = list(_fields(json.loads(content)))
            return record, fields, []
        # Preserve statements with their line numbers. Do not reproduce tables.
        lines = []
        for number, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if line and not line.startswith(("|", "!", "```")) and len(line) <= 900:
                lines.append({"line": number, "text": line})
        return record, [], lines
    except (UnicodeError, ValueError, OSError) as error:
        record["read_status"] = "unreadable_metadata"
        record["read_error"] = type(error).__name__
        return record, [], []


def _inspect(directory: Path, root: Path, origin: str, identifier: str):
    artifacts, semantic, text = [], [], []
    for path, kind in _artifacts(directory):
        artifact, fields, lines = _read_artifact(path, kind, root)
        artifacts.append(artifact)
        semantic.extend({**field, "source": artifact["path"]} for field in fields)
        text.extend({**line, "source": artifact["path"], "kind": kind} for line in lines)
    statuses = [field for field in semantic if field["key"].split(".")[-1].lower() in STATUS_KEYS]
    # Status statements in prose are retained as assertions, not normalized to COMPLETE.
    prose_status = [line for line in text if re.search(
        r"(?:状态|阶段|status|stage)\s*[:：=]|(?:全部|均|已)完成|停止|暂停|失败|not implemented|complete[d]?\b",
        line["text"], re.I)]
    role_keys = {"experiment_role", "evidence_role"}
    role_fields = [field for field in semantic if field["key"].split(".")[-1].lower() in role_keys]
    category, category_basis = "uncertain", []
    # Only explicit use statements count. Directory names are deliberately absent.
    patterns = [
        ("historical", r"^(?:historical|legacy|archived)$|^#{1,4}\s*历史(?:归档|实验)|(?:本实验|本目录|本轮).{0,20}(?:历史归档|历史复现)"),
        ("engineering", r"^(?:engineering|smoke)$|(?:本实验|本轮|当前).{0,25}(?:仅为|属于|用于).{0,12}(?:工程|冒烟|四更新)|^#{1,4}\s*(?:工程验证|冒烟测试)"),
        ("diagnostic", r"^(?:diagnostic|diagnosis|probe)$|(?:本实验|本轮|本报告).{0,25}(?:诊断|探针)|^#{1,4}\s*[^\n]{0,50}(?:诊断|探针)"),
        ("formal", r"^formal$|(?:本实验|本轮|实验角色).{0,15}(?:正式配对实验|正式实验)|^#{1,4}\s*正式(?:实验|协议|配对)"),
    ]
    candidates = [{"text": f["value"], "source": f["source"], "key": f["key"]} for f in role_fields]
    candidates += [line for line in text if line["kind"] in {"documentation", "report"}]
    for name, pattern in patterns:
        evidence = [line for line in candidates if re.search(pattern, line["text"], re.I)]
        if evidence:
            category, category_basis = name, evidence[:5]
            break
    goals = [field for field in semantic if field["key"].split(".")[-1].lower() in {"objective", "purpose", "goal", "description"}]
    goal_lines = [line for line in text if re.search(r"^#{1,3}\s|^(?:目标|目的|任务)\s*[:：]", line["text"])]
    goal = goals[0] if goals else (goal_lines[0] if goal_lines else None)
    scope = [field for field in semantic if field["key"].split(".")[-1].lower() not in STATUS_KEYS | {"objective", "purpose", "goal", "description", "experiment_role", "evidence_role"}]
    scope += [line for line in text if re.search(r"已暴露|独立|回看|历史|确认站|C2019|D2020|PWV|IWV|TCWV|信息条件|口径", line["text"], re.I)]
    missing = []
    if not goal:
        missing.append("目标未从已读元数据中定位；按原目录保留。")
    if not any(a["kind"] == "protocol" or Path(a["path"]).stem.lower() in {"protocol", "plan", "experiment_lock"} for a in artifacts):
        missing.append("未在扫描范围找到协议/计划/锁文件；可能位于目录外。")
    if not statuses and not prose_status:
        missing.append("无明确状态证据；不由目录名称或指标文件推断完成。")
    if not any(a["kind"] == "report" for a in artifacts):
        missing.append("未在扫描范围找到原结果报告。")
    if not scope:
        missing.append("历史证据范围、数据角色及暴露条件未从元数据中定位，需回到原协议核查。")
    if category == "uncertain":
        missing.append("未找到明确的实验用途声明，类别暂记 uncertain。")
    return {
        "id": identifier, "name": directory.name, "origin": origin,
        "original_path": _relative(directory, root), "category": category,
        "category_basis": category_basis, "goal": goal,
        "status": "reported_states" if statuses else ("reported_prose" if prose_status else "no_evidence"),
        "status_evidence": statuses, "prose_status_evidence": prose_status[:12],
        "scope_evidence": scope[:15], "source_artifacts": artifacts, "missing_evidence": missing,
    }


def _evidence_line(item: dict, root: Path, document: Path) -> str:
    location = _link(root / item["source"], document, item["source"])
    label = f"第{item['line']}行" if "line" in item else item.get("key", "元数据")
    value = item.get("text", item.get("value", ""))
    # Source-relative inline links belong to the original report. Keep their
    # labels here and make the explicit source link the single navigation target.
    value = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", value)
    return f"- {location}，{label}：{value}"


def _write_entry(entry: dict, root: Path, output: Path):
    document = output / "legacy" / entry["id"] / "README.md"
    document.parent.mkdir(parents=True, exist_ok=True)
    source = root / entry["original_path"]
    lines = [f"# {entry['name']}", "", f"当前公开来源：`{entry['original_path']}`；{_link(source, document, '来源目录')}。", "",
             f"历史路径：`{entry.get('historical_path') or '迁移映射中未定位'}`。", "",
             f"用途类别：`{entry['category']}`。状态索引：`{entry['status']}`。状态表示来源中的声明，各子运行分别列出。", "", "## 目标", ""]
    lines.append(_evidence_line(entry["goal"], root, document) if entry["goal"] else "未从已读元数据中定位。")
    for title, field in [("类别依据", "category_basis"), ("状态依据", "status_evidence"),
                         ("报告中的阶段声明", "prose_status_evidence"), ("历史证据范围与信息条件", "scope_evidence")]:
        lines += ["", f"## {title}", ""]
        lines += [_evidence_line(item, root, document) for item in entry[field]] or ["未定位到明确证据。"]
    lines += ["", "## 原始协议、结果与报告", ""]
    lines += [f"- {_link(root / item['path'], document, item['path'])}（{item['kind']}；{item['read_status']}）" for item in entry["source_artifacts"]] or ["扫描范围内未找到可索引元数据。"]
    lines += ["", "## 待核查项", ""]
    lines += [f"- {item}" for item in entry["missing_evidence"]] or ["已定位基本导航证据；科学结论以原报告、协议与审阅为准。"]
    lines += ["", "本页整理来源位置和原文声明，不重算或重写原始指标。扫描上限为5层、单文件512 KiB；数据、标签、权重和数组目录不读取。", ""]
    document.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _add_history(entries: list[dict], root: Path):
    mapping_file = root / "maintenance/paper_restructure_20261126/path_mapping.json"
    paths = json.loads(mapping_file.read_text(encoding="utf-8"))["paths"] if mapping_file.is_file() else {}
    inverse = {new: old for old, new in paths.items()}
    for entry in entries:
        prefix = entry["original_path"] + "/"
        candidates = []
        for current, old in inverse.items():
            if current.startswith(prefix):
                suffix = Path(current[len(prefix):]).parts
                candidates.append(Path(*Path(old).parts[:-len(suffix)]).as_posix())
        entry["historical_path"] = Counter(candidates).most_common(1)[0][0] if candidates else None
        entry["historical_path_basis"] = "maintenance/paper_restructure_20261126/path_mapping.json" if candidates else None
        for artifact in entry["source_artifacts"]:
            artifact["historical_path"] = inverse.get(artifact["path"])


def build_catalog(root: Path, output: Path, source_engine: Path | None = None) -> dict:
    """Write ``output/index.json``, README and legacy/<source-name>/README.md.

    Relative arguments are resolved against root. source_engine allows a migrated
    checkout while all emitted source paths remain relative to the project root.
    No source file is modified and no experiment code is imported or executed.
    """
    root = Path(root).resolve()
    output = (root / output).resolve() if not Path(output).is_absolute() else Path(output).resolve()
    engine = Path(source_engine) if source_engine is not None else (Path("research/engine") if (root / "research/engine").is_dir() else OLD_ENGINE)
    engine = (root / engine).resolve() if not engine.is_absolute() else engine.resolve()
    public_results = root / "experiments/us"
    results = public_results if public_results.is_dir() else engine / "results"
    if not results.is_dir() or _is_link(results):
        raise FileNotFoundError(f"Public experiment metadata directory missing: {results}")
    source_dirs = [(path, "us") for path in sorted(results.iterdir()) if path.is_dir() and not _is_link(path)]
    campaigns = root / "research"
    if campaigns.is_dir():
        for path in sorted(campaigns.iterdir()):
            if path.is_dir() and not _is_link(path) and path.name not in {"engine", "cleanup_20261123", "__pycache__"}:
                if any(path.glob("*.md")):
                    source_dirs.append((path, "research_campaign"))
    if any(output == path.resolve() or path.resolve() in output.parents for path, _ in source_dirs):
        raise ValueError("Catalog output must be outside source experiment directories")
    counts = Counter(path.name for path, _ in source_dirs)
    entries = []
    for path, origin in source_dirs:
        identifier = path.name if counts[path.name] == 1 else f"{origin}__{path.name}"
        entries.append(_inspect(path, root, origin, identifier))
    _add_history(entries, root)
    index = {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_engine": _relative(engine, root), "source_results": _relative(results, root), "experiment_count": len(entries),
        "category_counts": dict(Counter(e["category"] for e in entries)),
        "policy": {"numeric_results": "source links only; not copied or recomputed", "labels_arrays_weights_read": False,
                   "max_metadata_bytes": MAX_METADATA_BYTES, "max_depth": MAX_DEPTH,
                   "classification": "explicit content evidence; uncertain by default", "status": "source assertions; not independent verification"},
        "experiments": entries,
    }
    output.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        _write_entry(entry, root, output)
    (output / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = ["# 现有实验与证据索引", "", f"共 {len(entries)} 个原目录。原始协议、报告及数值保留在原位置；本索引记录来源及待核查项。", "",
             "用途类别由明确元数据或原文声明支持；证据不足记 uncertain。状态不由目录名、指标存在或文件时间推断。各子运行状态可以不同，历史计划和最终报告的声明分别保留。", "",
             "本次仅索引最多5层、每份不超过512 KiB的选定 Markdown/JSON 元数据，未读取站点标签、NPY/NPZ、权重或原始观测。未纳入的外部锁、超限元数据和更深层文件需按原项目追溯。", "",
             "| 原目录 | 来源 | 用途类别 | 状态证据 |", "|---|---|---|---|"]
    for entry in entries:
        target = quote(f"legacy/{entry['id']}/README.md", safe="/._-")
        lines.append(f"| [{entry['name']}]({target}) | {entry['origin']} | {entry['category']} | {entry['status']} |")
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments"))
    parser.add_argument("--source-engine", type=Path)
    args = parser.parse_args()
    result = build_catalog(args.root, args.output, args.source_engine)
    print(json.dumps({"experiment_count": result["experiment_count"], "category_counts": result["category_counts"]}, ensure_ascii=False))
