"""Causal GT-free replay of the deployable measured-first selector."""

from __future__ import annotations

import argparse, csv, hashlib, json, os, statistics, time
from pathlib import Path

import multiview_best_view_measured_first_v1 as method
import replay_view_ac_common_frame_engineering_v1 as replay_check


FIELDS = ["repeat_index","sequence_index","phase","canonical_joint","valid","x_m","y_m","z_m","selected_view","selected_source","selected_reliability","score_a","score_c","method_version"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))


def load_replay(path):
    validated=replay_check.validate_replay_directory(Path(path).resolve(),"deployment")
    selected={name:{} for name in ("raw_measured","k2_guarded","k4_inferred")}
    for row in rows(validated["estimates_path"]):
        if row["method"] in selected: selected[row["method"]][(int(row["sequence_index"]),row["canonical_joint"])]=row
    return selected


def point(row):
    return (float(row["x_m"]),float(row["y_m"]),float(row["z_m"])) if row["valid"]=="1" else None


def depth(row):
    return float(row["depth_m"]) if row["valid"]=="1" and row["depth_m"].strip() else None


def view_input(replay,sequence):
    torso=tuple(depth(replay["raw_measured"][(sequence,joint)]) for joint in method.TORSO_JOINTS)
    result={}
    for joint in method.ARM_JOINTS:
        raw=replay["raw_measured"][(sequence,joint)]
        k4=replay["k4_inferred"][(sequence,joint)]
        result[joint]={
            "k2_point":point(replay["k2_guarded"][(sequence,joint)]),
            "k4_point":point(k4),
            "reliability":method.depth_reliability(depth(raw),torso),
            "visibility":float(k4["visibility"] or 0.0),
        }
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); args=p.parse_args()
    protocol=read_json(args.protocol.resolve()); output=args.output_dir.resolve()
    if output.exists(): raise RuntimeError("Refusing to overwrite deployment replay.")
    if protocol.get("status")!="frozen_before_deployment_replay" or protocol["registered_sha256"].get("deployment_replay")!=sha(Path(__file__).resolve()): raise RuntimeError("Deployment replay protocol mismatch.")
    paths=protocol["registered_paths"]; output_rows=[]; timings=[]
    for repeat in (1,2,3):
        a=load_replay(paths[f"rep_{repeat:02d}_view_a_replay"]); c=load_replay(paths[f"rep_{repeat:02d}_view_c_replay"])
        extrinsics=read_json(paths[f"rep_{repeat:02d}_extrinsics"])
        if extrinsics["uses_skeleton_gt"] is not False: raise RuntimeError("GT-derived extrinsics prohibited.")
        transform=extrinsics["relative_transforms"]["a_from_c"]
        for sequence in range(240):
            start=time.perf_counter_ns()
            arm_a=method.build_view_arm(view_input(a,sequence))
            arm_c=method.build_view_arm(view_input(c,sequence),transform)
            selected=method.select_arm(arm_a,arm_c)
            timings.append((time.perf_counter_ns()-start)/1e6)
            for joint in method.ARM_JOINTS:
                item=selected["joints"][joint]; value=item["point_a"]
                output_rows.append({"repeat_index":repeat,"sequence_index":sequence,"phase":"active" if 60<=sequence<=89 or 150<=sequence<=179 else "inactive","canonical_joint":joint,"valid":int(value is not None),"x_m":value[0] if value else "","y_m":value[1] if value else "","z_m":value[2] if value else "","selected_view":selected["selected_view"],"selected_source":item["source"],"selected_reliability":item["reliability"],"score_a":json.dumps(selected["score_a"],separators=(",",":")),"score_c":json.dumps(selected["score_c"],separators=(",",":")),"method_version":method.METHOD_VERSION})
    output.mkdir(parents=True,exist_ok=False); csv_path=output/"deployment_best_view_rows.csv"
    with csv_path.open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,lineterminator="\n"); w.writeheader(); w.writerows(output_rows)
    state={"schema_version":1,"status":"complete","classification":"gt_free_deployment_equivalence_replay","gt_files_opened":0,"frame_count":720,"row_count":len(output_rows),"mean_runtime_ms":statistics.fmean(timings),"p95_runtime_ms":sorted(timings)[int(0.95*(len(timings)-1))],"max_runtime_ms":max(timings),"rows_sha256":sha(csv_path),"protocol_sha256":sha(args.protocol.resolve()),"method_source_sha256":sha(Path(method.__file__).resolve())}
    (output/"deployment_replay_state.json").write_text(json.dumps(state,indent=2)+"\n",encoding="utf-8"); print(json.dumps(state,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
