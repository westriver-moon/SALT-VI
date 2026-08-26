import csv
import json
import re
from datetime import datetime
from pathlib import Path


EXPERIMENT_ROOT = Path("/home/cgv841/ybj/SALT-VI/experiments/stageb-fusion-ablation")
SUMMARY_MD = EXPERIMENT_ROOT / "summary.md"
SUMMARY_CSV = EXPERIMENT_ROOT / "summary.csv"
TIME_RE = re.compile(r"Time:\s*([0-9:-]+);")
EPOCH_RE = re.compile(r"Time:\s*([0-9:-]+);\s*Epoch:\s*(\d+);")
BEST_RE = re.compile(
    r"Best Fusion_RGB mINP:\s*([0-9.]+), Best mAP:\s*([0-9.]+), Best Rank1:\s*([0-9.]+)"
)

BASELINES = [
    {
        "experiment": "E1 no_sff + add",
        "Feat_Filter": False,
        "fusion_way": "add",
        "pa": "",
        "best_epoch": 13,
        "Rank-1": 0.81767,
        "mAP": 0.78509,
        "mINP": 0.67427,
        "best_model_path": "/home/cgv841/ybj/SALT-VI/experiments/stageb-vit-no-sff/results/train/sysu/FV/Baseline_train[RGB_IR_Text]_joint[uni]_Blip_add_id,wrt_Fix_Visual/models/model_Fusion_13.pth",
        "training_time": "",
        "status": "succeeded",
        "log_path": "/home/cgv841/ybj/SALT-VI/logs/raw/experiments/stageb-vit-source_core-no-sff/results/logs/train.log",
    },
    {
        "experiment": "E2 sff + add",
        "Feat_Filter": True,
        "fusion_way": "add",
        "pa": "",
        "best_epoch": 13,
        "Rank-1": 0.80297,
        "mAP": 0.77435,
        "mINP": 0.66179,
        "best_model_path": "/home/cgv841/ybj/SALT-VI/experiments/stageb-vit-sff/results/train/sysu/FV_Filter/Baseline_train[RGB_IR_Text]_joint[uni]_Blip_add_id,wrt_Fix_Visual_Filtered/models/model_Fusion_13.pth",
        "training_time": "",
        "status": "succeeded",
        "log_path": "/home/cgv841/ybj/SALT-VI/logs/raw/experiments/stageb-vit-source_core-sff/results/logs/train.log",
    },
]

EXPERIMENTS = [
    {
        "experiment": "E3 no_sff + norm_add",
        "experiment_name": "e3_no_sff_norm_add",
        "Feat_Filter": False,
        "fusion_way": "norm_add",
        "pa": "",
    },
    {
        "experiment": "E4 no_sff + parameter_add_pa05",
        "experiment_name": "e4_no_sff_parameter_add_pa05",
        "Feat_Filter": False,
        "fusion_way": "parameter_add",
        "pa": 0.5,
    },
    {
        "experiment": "E5 no_sff + cross_attention",
        "experiment_name": "e5_no_sff_cross_attention",
        "Feat_Filter": False,
        "fusion_way": "cross_attention",
        "pa": "",
    },
]


def read_json(path):
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_log_time(value):
    return datetime.strptime(value, "%Y-%m-%d-%H:%M:%S")


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_training_time_from_log(log_path):
    path = Path(log_path)
    if not path.is_file():
        return ""
    timestamps = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TIME_RE.search(line)
        if match:
            timestamps.append(parse_log_time(match.group(1)))
    if len(timestamps) < 2:
        return ""
    return format_duration((max(timestamps) - min(timestamps)).total_seconds())


def parse_best_from_log(log_path):
    path = Path(log_path)
    if not path.is_file():
        return {
            "best_epoch": None,
            "best_rank1": None,
            "best_map": None,
            "best_minp": None,
        }

    best_epoch = None
    best_rank1 = None
    best_map = None
    best_minp = None
    current_epoch = None
    pending_new_best = False

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(2))
            continue

        if "New Best" in line:
            pending_new_best = True
            continue

        best_match = BEST_RE.search(line)
        if best_match and (pending_new_best or best_epoch is None):
            best_minp = float(best_match.group(1))
            best_map = float(best_match.group(2))
            best_rank1 = float(best_match.group(3))
            best_epoch = current_epoch
            pending_new_best = False

    return {
        "best_epoch": best_epoch,
        "best_rank1": best_rank1,
        "best_map": best_map,
        "best_minp": best_minp,
    }


def resolve_best_model_path(experiment_name, best_epoch, status_best_model_path):
    if status_best_model_path:
        path = Path(status_best_model_path)
        if path.is_file():
            return str(path)

    if best_epoch is None:
        return ""

    matches = sorted((EXPERIMENT_ROOT / experiment_name).rglob(f"model_Fusion_{best_epoch}.pth"))
    if matches:
        return str(matches[-1])
    return ""


def format_metric(value):
    if value in (None, ""):
        return ""
    return f"{float(value):.5f}"


def build_rows():
    rows = []
    for baseline in BASELINES:
        row = dict(baseline)
        row["training_time"] = row["training_time"] or parse_training_time_from_log(row["log_path"])
        row.pop("log_path", None)
        rows.append(row)

    for meta in EXPERIMENTS:
        payload = read_json(EXPERIMENT_ROOT / meta["experiment_name"] / "status.json") or {}
        log_path = EXPERIMENT_ROOT / meta["experiment_name"] / "launcher.log"
        parsed = parse_best_from_log(log_path)
        best_epoch = payload.get("best_epoch")
        best_rank1 = payload.get("best_rank1")
        best_map = payload.get("best_map")
        best_minp = payload.get("best_minp")

        if parsed["best_epoch"] is not None:
            best_epoch = parsed["best_epoch"]
            best_rank1 = parsed["best_rank1"]
            best_map = parsed["best_map"]
            best_minp = parsed["best_minp"]

        rows.append(
            {
                "experiment": meta["experiment"],
                "Feat_Filter": meta["Feat_Filter"],
                "fusion_way": meta["fusion_way"],
                "pa": meta["pa"],
                "best_epoch": best_epoch,
                "Rank-1": best_rank1,
                "mAP": best_map,
                "mINP": best_minp,
                "best_model_path": resolve_best_model_path(
                    meta["experiment_name"], best_epoch, payload.get("best_model_path")
                ),
                "training_time": payload.get("training_time") or parse_training_time_from_log(log_path),
                "status": payload.get("status", "pending"),
            }
        )
    return rows


def sort_key(row):
    if row["mAP"] is None:
        return (1, 0.0)
    return (0, -float(row["mAP"]))


def write_csv(rows):
    fieldnames = [
        "experiment",
        "Feat_Filter",
        "fusion_way",
        "pa",
        "best_epoch",
        "Rank-1",
        "mAP",
        "mINP",
        "best_model_path",
        "training_time",
        "status",
    ]
    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(rows):
    e1 = next(row for row in rows if row["experiment"] == "E1 no_sff + add")
    e2 = next(row for row in rows if row["experiment"] == "E2 sff + add")
    successful = [row for row in rows if row["mAP"] is not None]
    successful.sort(key=lambda row: float(row["mAP"]), reverse=True)
    best_row = successful[0] if successful else None

    best_fusion_way = best_row["fusion_way"] if best_row else "unknown"
    beats_e1 = bool(best_row and float(best_row["mAP"]) > float(e1["mAP"]))
    beats_e2 = bool(best_row and float(best_row["mAP"]) > float(e2["mAP"]))
    suggest_stage2 = "yes" if beats_e1 else "no"

    lines = [
        "# Stage B Fusion Ablation Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| experiment | Feat_Filter | fusion_way | pa | best_epoch | Rank-1 | mAP | mINP | best_model_path | training_time | status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in rows:
        lines.append(
            "| {experiment} | {Feat_Filter} | {fusion_way} | {pa} | {best_epoch} | {rank1} | {mapv} | {minp} | {best_model_path} | {training_time} | {status} |".format(
                experiment=row["experiment"],
                Feat_Filter=row["Feat_Filter"],
                fusion_way=row["fusion_way"],
                pa=row["pa"],
                best_epoch=row["best_epoch"] if row["best_epoch"] is not None else "",
                rank1=format_metric(row["Rank-1"]),
                mapv=format_metric(row["mAP"]),
                minp=format_metric(row["mINP"]),
                best_model_path=row["best_model_path"],
                training_time=row["training_time"],
                status=row["status"],
            )
        )

    lines.extend(
        [
            "",
            f"- 当前最佳 fusion_way: {best_fusion_way}",
            f"- 是否超过 E1 no_sff + add baseline: {'yes' if beats_e1 else 'no'}",
            f"- 是否超过 E2 sff + add baseline: {'yes' if beats_e2 else 'no'}",
            f"- 是否建议进入第二阶段实验: {suggest_stage2}",
            "",
            "排序规则: 按 mAP 从高到低。",
        ]
    )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    rows.sort(key=sort_key)
    write_csv(rows)
    write_md(rows)
    print(str(SUMMARY_MD))
    print(str(SUMMARY_CSV))


if __name__ == "__main__":
    main()
