"""Create a training view; filter excluded episodes and share source videos."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

TASK_FILE = Path(__file__).with_name("task_instruction.txt")
DEFAULT_TASK = TASK_FILE.read_text(encoding="utf-8").strip()
VIDEO_KEYS = ["left_wrist", "left_top", "right_wrist"]


def prepare(source, output, task, skip_first=0):
    task = task.strip()
    if not task:
        raise ValueError("Task instruction must not be empty")
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and training view must be separate directories")
    info = json.loads((source / "meta/info.json").read_text())
    if not 0 <= skip_first < info["total_episodes"]:
        raise ValueError("skip_first must leave at least one episode and cannot be negative")
    modality = json.loads((source / "meta/modality.json").read_text())
    tasks = pd.read_parquet(source / "meta/tasks.parquet")
    if info["codebase_version"] != "v3.0" or len(tasks) != 1:
        raise ValueError("This preparation supports the inspected single-task v3 dataset only")
    if info["splits"] != {"train": f"0:{info['total_episodes']}"}:
        raise ValueError("Expected every episode in the training split")
    old_task = tasks.index[0]
    episode_files = sorted((source / "meta/episodes").rglob("*.parquet"))
    tables = [pd.read_parquet(f) for f in episode_files]
    indices = [int(i) for table in tables for i in table["episode_index"]]
    if sorted(indices) != list(range(info["total_episodes"])):
        raise ValueError("Episode indices/count do not match info.json")
    if sum(int(n) for table in tables for n in table["length"]) != info["total_frames"]:
        raise ValueError("Episode row totals do not match info.json")
    if any(list(names) != [old_task] for table in tables for names in table["tasks"]):
        raise ValueError("Mixed/unknown task labels; cannot replace them with one instruction")
    meta_files = [source / "meta" / n for n in ["info.json", "modality.json", "tasks.parquet"]]
    meta_files += episode_files
    provenance = {
        "source": str(source), "task": task, "video_keys": VIDEO_KEYS,
        "source_metadata_sha256": {
            str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in meta_files
        },
    }
    if skip_first:
        provenance["excluded_episode_indices"] = list(range(skip_first))
        tables = [table.loc[table["episode_index"] >= skip_first].copy() for table in tables]
        info["splits"] = {"train": f"{skip_first}:{info['total_episodes']}"}
        info["total_episodes"] -= skip_first
        info["total_frames"] = sum(int(n) for table in tables for n in table["length"])
    if output.exists():
        marker = output / "baseline_view.json"
        if not marker.exists() or json.loads(marker.read_text()) != provenance:
            raise FileExistsError(f"Refusing to overwrite existing/different training view: {output}")
        for name in ["data", "videos"]:
            if name == "data" and skip_first:
                if not (output / name).is_dir() or (output / name).is_symlink():
                    raise ValueError("Filtered training data must be a separate directory")
                continue
            if not (output / name).is_symlink() or (output / name).resolve() != source / name:
                raise ValueError(f"Training view link changed: {name}")
        print(f"Existing baseline view: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        shutil.copytree(source / "meta", temporary / "meta")
        for name in ["data", "videos"]:
            if name == "data" and skip_first:
                # Stats scans all Parquet rows, so metadata filtering alone would
                # still include excluded episodes in normalization statistics.
                row_count = 0
                for path in sorted((source / "data").rglob("*.parquet")):
                    rows = pq.read_table(path)
                    rows = rows.filter(pc.greater_equal(rows["episode_index"], skip_first))
                    if not len(rows):
                        continue
                    destination = temporary / path.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(rows, destination)
                    row_count += len(rows)
                if row_count != info["total_frames"]:
                    raise ValueError("Filtered data row count does not match episode metadata")
                continue
            (temporary / name).symlink_to(source / name, target_is_directory=True)
        # The stock stats generator scans all float features, including unused
        # nested tactile arrays that it cannot process. Do not declare those
        # unused features in this baseline's metadata. Source files are unchanged.
        selected_images = {f"observation.images.{key}" for key in VIDEO_KEYS}
        info["features"] = {
            key: value for key, value in info["features"].items()
            if key != "observation.tactile_f6"
            and (not key.startswith("observation.images.") or key in selected_images)
        }
        modality["video"] = {key: modality["video"][key] for key in VIDEO_KEYS}
        (temporary / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
        (temporary / "meta/modality.json").write_text(json.dumps(modality, indent=2))
        tasks.index = pd.Index([task], name=tasks.index.name)
        tasks.to_parquet(temporary / "meta/tasks.parquet")
        for file, table in zip(episode_files, tables):
            table["tasks"] = [[task] for _ in range(len(table))]
            table.to_parquet(temporary / file.relative_to(source), index=False)
        for name in ["stats.json", "relative_stats.json"]:
            (temporary / "meta" / name).unlink(missing_ok=True)
        (temporary / "baseline_view.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(f"Prepared baseline view: {output}")
    print(f"Retained {info['total_episodes']} episodes / {info['total_frames']} rows; excluded first {skip_first} episodes")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--skip-first", type=int, default=0)
    args = parser.parse_args()
    prepare(args.source, args.output, args.task, skip_first=args.skip_first)
