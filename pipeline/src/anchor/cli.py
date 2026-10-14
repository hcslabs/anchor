"""``anchor`` command-line entry point.

    anchor probe clip.mp4          probe a file, print its metadata + content ID
    anchor gates clip.mp4          run the open gate cascade, print the verdict
    anchor stages                  list the seven stages and their implementation status
    anchor retarget-g1 ...         HCS human-landmark → KHI Unitree-G1 retargeting
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anchor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("probe", help="probe a video file")
    sp.add_argument("path")
    sg = sub.add_parser("gates", help="run the open quality-gate cascade")
    sg.add_argument("path")
    sub.add_parser("stages", help="list pipeline stages")
    sr = sub.add_parser("retarget-g1", help="retarget HCS human landmarks to the KHI G1 URDF")
    sr.add_argument("--human-motion", required=True, help="Anchor human_motion_v1 JSON")
    sr.add_argument("--urdf", required=True, help="KHI G1 29-DoF URDF")
    sr.add_argument("--output", required=True, help="output Anchor G1 trajectory JSON")
    sr.add_argument(
        "--profile", choices=("full_body", "khi_tabletop_right"), default="full_body"
    )
    sre = sub.add_parser(
        "retarget-embodiment",
        help="retarget human landmarks using a versioned robot embodiment profile",
    )
    sre.add_argument("--human-motion", required=True)
    sre.add_argument("--urdf", required=True)
    sre.add_argument("--embodiment", required=True)
    sre.add_argument("--profile", required=True)
    sre.add_argument("--output", required=True)
    se = sub.add_parser("extract-human-pose", help="derive a screening 3-D skeleton from an approved human video")
    se.add_argument("--video", required=True)
    se.add_argument("--model", required=True, help="MediaPipe Pose Landmarker task file")
    se.add_argument("--output", required=True)
    se.add_argument("--sample-fps", type=float, default=5.0)
    sx = sub.add_parser("export-khi-g1", help="export Anchor G1 motion to KHI Isaac Lab joblib")
    sx.add_argument("--trajectory", required=True)
    sx.add_argument("--output", required=True)
    sx.add_argument("--root-height-m", type=float, default=0.793)
    sv = sub.add_parser("validate-khi-g1", help="validate a KHI Isaac Lab robot reference")
    sv.add_argument("path")
    so = sub.add_parser("export-khi-object", help="create a free-object companion for KHI replay")
    so.add_argument("--robot-reference", required=True)
    so.add_argument("--output", required=True)
    so.add_argument(
        "--relative-position-m", type=float, nargs=3,
        default=(0.487, -0.180, -0.089), metavar=("X", "Y", "Z"),
    )
    svp = sub.add_parser("validate-khi-pair", help="validate robot/object files against KHI replay fields")
    svp.add_argument("--robot-reference", required=True)
    svp.add_argument("--object-reference", required=True)
    sk = sub.add_parser("proof-khi", help="build a no-copy Anchor → KHI proof manifest")
    sk.add_argument("--khi-root", required=True)
    sk.add_argument("--human-video", required=True)
    sk.add_argument("--human-motion", required=True)
    sk.add_argument("--g1-trajectory", required=True)
    sk.add_argument("--khi-reference", required=True)
    sk.add_argument("--khi-object-reference", required=True)
    sk.add_argument("--output", required=True)
    spv = sub.add_parser("proof-verify", help="verify a hash-bound Anchor → KHI proof")
    spv.add_argument("manifest")
    spv.add_argument("--receipt")
    scl = sub.add_parser(
        "closed-loop-khi",
        help="run bounded human-video retargeting search and emit KHI artifacts after gates",
    )
    scl.add_argument("--video", action="append", required=True)
    scl.add_argument("--model", required=True)
    scl.add_argument("--urdf", required=True)
    scl.add_argument("--embodiment", required=True)
    scl.add_argument("--profile", default="khi_tabletop_right")
    scl.add_argument("--output-dir", required=True)
    scl.add_argument("--sample-fps", type=float, default=5.0)
    scl.add_argument("--feedback-json")

    args = p.parse_args(argv)

    if args.cmd == "probe":
        from anchor.ingest import ingest
        ep = ingest(args.path)
        print(ep.to_json())
        return 0

    if args.cmd == "gates":
        from anchor.ingest import probe, ingest
        from anchor.quality import GateCascade, default_open_gates
        ep = ingest(args.path)
        ok = GateCascade(default_open_gates()).run(probe(args.path), ep)
        print(ep.to_json())
        return 0 if ok else 1

    if args.cmd == "stages":
        from anchor.pipeline.stages import STAGE_ORDER
        status = {
            "intake": "working/open",
            "trust": "closed/interface",
            "gates": "working/open",
            "enrich": "partial/open+stubs",
            "retarget": "working/HCS-G1+closed methods",
            "package": "stub",
            "serve": "stub",
        }
        for i, name in enumerate(STAGE_ORDER, 1):
            mark = status[name]
            print(f"  {i:02d}  {name:<10s} [{mark}]")
        return 0

    if args.cmd == "retarget-g1":
        from anchor.retarget import retarget_file
        result = retarget_file(
            args.human_motion, args.urdf, args.output, profile=args.profile
        )
        print(json.dumps({
            "output": args.output,
            "kind": result["kind"],
            "uses_premade_robot_motion": result["uses_premade_robot_motion"],
            "quality": result["quality"],
        }, indent=2))
        return 0

    if args.cmd == "retarget-embodiment":
        from anchor.retarget import retarget_embodiment_file
        result = retarget_embodiment_file(
            args.human_motion,
            args.urdf,
            args.embodiment,
            args.output,
            profile=args.profile,
        )
        print(json.dumps({
            "output": args.output,
            "kind": result["kind"],
            "target_robot": result["target_robot"],
            "uses_premade_robot_motion": result["uses_premade_robot_motion"],
            "quality": result["quality"],
        }, indent=2))
        return 0

    if args.cmd == "extract-human-pose":
        from anchor.enrich.human_pose import extract_human_motion
        result = extract_human_motion(
            args.video, args.model, args.output, sample_fps=args.sample_fps
        )
        print(json.dumps({
            "output": args.output,
            "source_video": result["source_video"],
            "quality": result["quality"],
        }, indent=2))
        return 0

    if args.cmd == "export-khi-g1":
        from anchor.integrations.khi_g1 import export_khi_robot_reference
        receipt = export_khi_robot_reference(
            args.trajectory, args.output, root_height_m=args.root_height_m
        )
        print(json.dumps(receipt, indent=2))
        return 0

    if args.cmd == "validate-khi-g1":
        from anchor.integrations.khi_g1 import validate_khi_robot_reference
        print(json.dumps(validate_khi_robot_reference(args.path), indent=2))
        return 0

    if args.cmd == "export-khi-object":
        from anchor.integrations.khi_g1 import export_khi_object_reference
        receipt = export_khi_object_reference(
            args.robot_reference,
            args.output,
            relative_position_m=tuple(args.relative_position_m),
        )
        print(json.dumps(receipt, indent=2))
        return 0

    if args.cmd == "validate-khi-pair":
        from anchor.integrations.khi_g1 import validate_khi_replay_pair
        result = validate_khi_replay_pair(args.robot_reference, args.object_reference)
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "proof-khi":
        from anchor.proof import build_khi_proof_manifest
        manifest = build_khi_proof_manifest(
            khi_root=args.khi_root,
            human_video=args.human_video,
            human_motion=args.human_motion,
            g1_trajectory=args.g1_trajectory,
            khi_reference=args.khi_reference,
            khi_object_reference=args.khi_object_reference,
            output_path=args.output,
        )
        print(json.dumps({
            "output": args.output,
            "proof_id": manifest["proof_id"],
            "artifact_count": len(manifest["artifacts"]),
            "disk_policy": manifest["disk_policy"],
        }, indent=2))
        return 0

    if args.cmd == "proof-verify":
        from anchor.proof import verify_khi_proof_manifest
        receipt = verify_khi_proof_manifest(args.manifest, args.receipt)
        print(json.dumps(receipt, indent=2))
        return 0 if receipt["technical_lineage_complete"] else 1

    if args.cmd == "closed-loop-khi":
        from anchor.agent.closed_loop import run_closed_loop
        receipt = run_closed_loop(
            videos=args.video,
            model_path=args.model,
            urdf_path=args.urdf,
            embodiment_path=args.embodiment,
            output_directory=args.output_dir,
            motion_profile=args.profile,
            sample_fps=args.sample_fps,
            feedback_path=args.feedback_json,
        )
        print(json.dumps({
            "output": str(args.output_dir),
            "technical_pass": receipt["technical_pass"],
            "physics_pass": receipt["physics_pass"],
            "physical_robot_commands_sent": receipt["physical_robot_commands_sent"],
            "selected_candidate": (
                receipt["selected_candidate"]["video_path"]
                if receipt["selected_candidate"] else None
            ),
        }, indent=2))
        return 0 if receipt["technical_pass"] else 1

    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:            # piping to head/grep is normal usage
        sys.stderr.close()
        sys.exit(0)
