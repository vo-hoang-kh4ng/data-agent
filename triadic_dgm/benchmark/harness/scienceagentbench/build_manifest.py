"""Build the ScienceAgentBench task manifest.

SAB programs are written against the repo-root layout: they read
`benchmark/datasets/<task_dir>/...` and write `pred_results/<output_fname>`, both
relative to the benchmark *parent*. To keep tasks isolated (so the blackboard
doesn't scan all 519 dataset files per task, and parallel runs don't cross-read)
we give each task its own workspace that reproduces just the path prefix it needs:
    <workspace>/benchmark/datasets/<task_dir>/...   (copied from the extracted data)
The agent runs with cwd=<workspace>, so its `benchmark/datasets/...` reads resolve.
Grading later copies the produced `pred_results/<output_fname>` up to the shared
data/sab/ root and runs the official eval_program with cwd=data/sab/.
"""
import json
import os
import re
import shutil
from typing import Dict, List

_TOP_DIR_RE = re.compile(r"\|?--\s+([^/|]+?)/\s*$")

# Heavy scientific deps that SAB gold programs often need but that are not always
# installed. torch/tensorflow/matplotlib are common enough to treat as available.
# A task whose gold program imports a missing heavy dep is "env-infeasible": the
# agent cannot reasonably reproduce the gold output locally, so counting it as a
# solver failure would understate the solver's true ability. We still RUN such
# tasks, but summarize() reports pass rates over the env-feasible subset too.
_HEAVY_DEPS = (
    "deepchem", "rdkit", "MDAnalysis", "mdtraj", "skimage", "mne", "geopandas",
    "obspy", "xarray", "netCDF4", "scanpy", "anndata", "dgl", "statsmodels",
    "pptx", "scholarly", "pysam", "Bio", "nglview", "ase", "pymatgen", "ecos",
    "ecospold", "osmnx", "pyTMG", "pmdarima", "formulaic",
)
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w]*)", re.M)

_available_heavy_cache = None


def _available_heavy() -> set:
    global _available_heavy_cache
    if _available_heavy_cache is None:
        import importlib
        _available_heavy_cache = set()
        for m in _HEAVY_DEPS:
            try:
                importlib.import_module(m)
                _available_heavy_cache.add(m)
            except Exception:  # noqa: BLE001
                pass
    return _available_heavy_cache


def _gold_program_deps(gold_path: str) -> list:
    """Heavy module names imported by the gold program (for env-feasibility)."""
    try:
        src = open(gold_path, encoding="utf-8", errors="replace").read()
    except Exception:  # noqa: BLE001
        return []
    mods = set(_IMPORT_RE.findall(src))
    return sorted(mods & set(_HEAVY_DEPS))


def _dataset_top_dirs(tree: str, src_path: str) -> List[str]:
    """Top-level dataset directory name(s) the task needs, from its folder tree."""
    dirs = []
    for line in (tree or "").splitlines():
        m = _TOP_DIR_RE.match(line.strip())
        if m:
            dirs.append(m.group(1).strip())
    if not dirs and src_path:
        base = src_path.strip("/").split("/")[-1]
        if base:
            dirs.append(base)
    # de-dup, preserve order
    seen, out = set(), []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def build_description(r: Dict, task_dirs: List[str]) -> str:
    tree = r.get("dataset_folder_tree", "")
    preview = r.get("dataset_preview", "")
    knowledge = r.get("domain_knowledge", "")
    out_fname = r.get("output_fname", "")
    parts = [
        "SCIENTIFIC DATA-ANALYSIS TASK (ScienceAgentBench).",
        f"Domain: {r.get('domain','')}",
        "",
        f"Task instruction: {r.get('task_inst','')}",
        "",
    ]
    if knowledge:
        parts += ["Domain knowledge (hints):", knowledge[:1500], ""]
    parts += [
        f"Dataset folder tree (files live under benchmark/datasets/<dir>/ in the current working directory):",
        tree,
        "",
    ]
    if preview:
        parts += ["Dataset preview (truncated):", preview[:2000], ""]
    parts += [
        "TARGET PROGRAMMING LANGUAGE IS PYTHON.",
        "Write a SELF-CONTAINED Python program that:",
        f"  1. loads the input data from 'benchmark/datasets/{task_dirs[0] if task_dirs else '<dir>'}/' "
        "(use the exact relative paths shown in the folder tree above),",
        f"  2. performs the analysis described, and",
        f"  3. SAVES the final result to '{out_fname}' (create the directory if needed).",
        "Do not rely on any data not in the folder tree. Print a short confirmation when done.",
    ]
    return "\n".join(parts)


def build_manifest(
    data_dir: str,
    workspaces_root: str = "",
    limit: int = -1,
    offset_id: int = 0,
    prefer_csv: bool = False,
    gradeable_only: bool = False,
    feasible_only: bool = False,
) -> List[Dict]:
    ann_path = os.path.join(data_dir, "annotations.jsonl")
    bench_root = os.path.join(data_dir, "benchmark")
    src_datasets = os.path.join(bench_root, "datasets")
    rows = [json.loads(l) for l in open(ann_path, encoding="utf-8") if l.strip()]

    workspaces_root = workspaces_root or os.path.join(data_dir, "workspaces")
    os.makedirs(workspaces_root, exist_ok=True)

    # Classify each task's eval script: plot tasks need the GPT-4o visual judge
    # (gpt4_visual_judge / encode_image) which is unavailable in a local, no-GPT-4o
    # environment; they cannot be graded locally. `gradeable_only` drops them so a
    # scaling run doesn't spend hours of agent time on tasks it can't score.
    def _eval_kind(r):
        es = os.path.join(bench_root, "eval_programs", r.get("eval_script_name", ""))
        if not os.path.exists(es):
            return "missing"
        try:
            src = open(es, encoding="utf-8", errors="replace").read().lower()
        except Exception:  # noqa: BLE001
            return "missing"
        if "gpt4_visual_judge" in src or "encode_image" in src or "visual_judge" in src:
            return "plot"
        return "csv" if "read_csv" in src else "other"

    # Optionally order CSV-gradeable tasks first (for smoke testing on locally-gradeable tasks).
    if prefer_csv:
        rows.sort(key=lambda r: (0 if _eval_kind(r) == "csv" else 1, r["instance_id"]))

    avail = _available_heavy()
    tasks: List[Dict] = []
    for r in rows:
        if r["instance_id"] < offset_id:
            continue
        kind = _eval_kind(r)
        if gradeable_only and kind in ("plot", "missing"):
            continue
        task_dirs = _dataset_top_dirs(r.get("dataset_folder_tree", ""), r.get("src_file_or_path", ""))
        ws = os.path.abspath(os.path.join(workspaces_root, str(r["instance_id"])))
        # seed workspace with the task's dataset dir(s) under benchmark/datasets/
        dst_datasets = os.path.join(ws, "benchmark", "datasets")
        if os.path.isdir(src_datasets):
            for d in task_dirs:
                sdir = os.path.join(src_datasets, d)
                if os.path.isdir(sdir):
                    ddir = os.path.join(dst_datasets, d)
                    if not os.path.exists(ddir):
                        try:
                            shutil.copytree(sdir, ddir)
                        except Exception as e:  # noqa: BLE001 - partial data shouldn't skip the task
                            print(f"[sab] warn: copy {d} failed: {e}")
        gold_path = os.path.join(bench_root, "gold_programs", r.get("gold_program_name", ""))
        needed = _gold_program_deps(gold_path)
        env_feasible = all(d in avail for d in needed)
        if feasible_only and not env_feasible:
            continue
        tasks.append({
            "task_id": f"sab-{r['instance_id']}",
            "instance_id": r["instance_id"],
            "description": build_description(r, task_dirs),
            "workspace": ws,
            "output_fname": r.get("output_fname", ""),
            "eval_script_name": r.get("eval_script_name", ""),
            "gold_program_name": r.get("gold_program_name", ""),
            "domain": r.get("domain", ""),
            "task_inst": r.get("task_inst", ""),
            "task_dirs": task_dirs,
            "needed_deps": needed,
            "env_feasible": env_feasible,
            "eval_kind": kind,
        })
        if limit >= 0 and len(tasks) >= limit:
            break
    return tasks


if __name__ == "__main__":
    import sys
    from triadic_dgm.benchmark.harness.scienceagentbench.prepare import prepare
    dd = prepare()
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ts = build_manifest(dd, limit=lim, prefer_csv=True)
    print(f"built {len(ts)} SAB tasks (prefer_csv); sample task_dirs={ts[0]['task_dirs']} output={ts[0]['output_fname']}")
