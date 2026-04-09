"""Check tool for task-maker target/depth validation."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Sequence
from hashlib import sha1
from pathlib import Path

from pydantic import Field

from alpha.shared.record_id_codec import format_record_id
from alpha.shared.target_test_details_schema import load_target_details_file
from alpha.shared.target_id_codec import (
    parse_target_id,
    repo_to_slug,
    target_id_to_slug,
)
from openhands.sdk import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


def _resolve_timeout_seconds() -> int:
    raw = os.getenv("RUN_TESTS_TIMEOUT")
    if raw is None or not raw.strip():
        raise ValueError("RUN_TESTS_TIMEOUT is missing")
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError("RUN_TESTS_TIMEOUT must be an integer") from exc
    if value <= 0:
        raise ValueError("RUN_TESTS_TIMEOUT must be > 0")
    return value


def _runtime_paths(target_id: str, depth: int) -> tuple[Path, Path]:
    repo, _ = parse_target_id(target_id)
    repo_slug = repo_to_slug(repo)
    target_id_slug = target_id_to_slug(target_id)
    run_tests_path = Path(f"/runtime/{repo_slug}/run_tests.py")
    check_depth_dir = Path(f"/output/records/{target_id_slug}/check/depth_{depth}")
    return run_tests_path, check_depth_dir


def _details_path(turn_dir: Path) -> Path:
    return turn_dir / "details.json"


def _save_root(target_id: str) -> Path:
    target_id_slug = target_id_to_slug(target_id)
    return Path(f"/output/records/{target_id_slug}/save")


def _depth_state_path(target_id: str) -> Path:
    target_id_slug = target_id_to_slug(target_id)
    return Path(f"/runtime/targets/{target_id_slug}/state.json")


def _candidate_depth(target_id: str) -> int:
    state_path = _depth_state_path(target_id)
    if not state_path.exists():
        raise ValueError(
            f"missing depth state: {state_path}. target conversation must initialize current_depth first"
        )
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid depth state json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid depth state json: root must be an object")
    payload_target_id = payload.get("target_id")
    if payload_target_id != target_id:
        raise ValueError("invalid depth state json: target_id mismatch")
    current_depth = payload.get("current_depth")
    if not isinstance(current_depth, int) or current_depth < 1:
        raise ValueError("invalid depth state json: current_depth must be int >= 1")
    return current_depth


def _baseline_json_path() -> Path:
    return Path("/output/baseline/baseline.json")


def _next_check_index(check_depth_dir: Path) -> int:
    max_index = 0
    for path in check_depth_dir.glob("turn_*"):
        if not path.is_dir():
            continue
        match = re.match(r"^turn_(\d+)$", path.name)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _parse_result_text_at_least_one_header(text: str) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Invalid result format: empty result text")

    section: str | None = None
    saw_passed_header = False
    saw_failed_header = False
    passed_paths: list[str] = []
    failed_paths: list[str] = []

    for line in lines:
        lower_line = line.lower()
        if lower_line == "passed test files:":
            section = "passed"
            saw_passed_header = True
            continue
        if lower_line == "failed test files:":
            section = "failed"
            saw_failed_header = True
            continue

        if section == "passed":
            if not line.startswith("/testbed/"):
                raise ValueError(
                    "Invalid result format: passed paths must start with /testbed/"
                )
            passed_paths.append(line)
            continue

        if section == "failed":
            if not line.startswith("/testbed/"):
                raise ValueError(
                    "Invalid result format: failed paths must start with /testbed/"
                )
            failed_paths.append(line)
            continue

        # Ignore pre-header logs.
        continue

    if not (saw_passed_header or saw_failed_header):
        raise ValueError(
            "Invalid result format: expected at least one of passed/failed headers"
        )
    passed_paths = sorted(set(passed_paths))
    failed_paths = sorted(set(failed_paths))
    overlap = sorted(set(passed_paths) & set(failed_paths))
    if overlap:
        raise ValueError(
            "Invalid result format: the same test file appears in both passed and "
            f"failed sections: {', '.join(overlap)}"
        )
    return passed_paths, failed_paths

def _load_target_details(
    details_path: Path,
    *,
    expected_target_test_file_path: str,
) -> dict[str, object] | None:
    return load_target_details_file(
        details_path,
        expected_target_test_file_path=expected_target_test_file_path,
    )


def _load_baseline_pass() -> list[str]:
    baseline_path = _baseline_json_path()
    if not baseline_path.exists():
        raise ValueError(
            f"missing baseline file: {baseline_path}. run baseline(depth=0) first"
        )

    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid baseline json: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("invalid baseline json: root must be an object")

    baseline_pass_raw = payload.get("baseline_pass")
    if not isinstance(baseline_pass_raw, list):
        raise ValueError("invalid baseline json: baseline_pass must be a list")

    baseline_pass: list[str] = []
    for item in baseline_pass_raw:
        if not isinstance(item, str) or not item.startswith("/testbed/"):
            raise ValueError(
                "invalid baseline json: each baseline_pass item must start with /testbed/"
            )
        baseline_pass.append(item)
    return sorted(set(baseline_pass))


def _compute_workspace_fingerprint() -> str:
    proc = subprocess.run(
        [
            "bash",
            "-lc",
            (
                "set -euo pipefail; "
                "git -C /testbed status --porcelain=v1 --untracked-files=all; "
                "git -C /testbed diff --binary HEAD"
            ),
        ],
        capture_output=True,
        text=False,
        check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"fingerprint command failed with returncode={proc.returncode}: {stderr}"
        )
    return sha1(proc.stdout or b"").hexdigest()


def _write_check_meta(
    *,
    meta_path: Path,
    target_id: str,
    record_id: str,
    repo: str,
    depth: int,
    round_index: int,
    result_path: Path,
    workspace_fingerprint: str,
    f2p: list[str],
    p2p: list[str],
    target_test_file_path: str,
    target_file_failed: bool,
    other_failed_test_files: list[str],
    details_path: Path | None = None,
    target_total_test_cases: int | None = None,
    target_passed_test_cases: int | None = None,
    target_failed_test_cases: int | None = None,
    target_test_case_pass_rate: float | None = None,
    target_passed_test_case_ids: list[str] | None = None,
    target_failed_test_case_ids: list[str] | None = None,
) -> None:
    payload = {
        "ok": True,
        "target_id": target_id,
        "record_id": record_id,
        "repo": repo,
        "depth": depth,
        "round": round_index,
        "result_path": str(result_path),
        "workspace_fingerprint": workspace_fingerprint,
        "f2p": f2p,
        "p2p": p2p,
        "target_test_file_path": target_test_file_path,
        "target_file_failed": target_file_failed,
        "other_failed_test_files": other_failed_test_files,
        "other_failed_test_file_count": len(other_failed_test_files),
    }
    if details_path is not None:
        payload["details_path"] = str(details_path)
    if target_total_test_cases is not None:
        payload["target_total_test_cases"] = target_total_test_cases
    if target_passed_test_cases is not None:
        payload["target_passed_test_cases"] = target_passed_test_cases
    if target_failed_test_cases is not None:
        payload["target_failed_test_cases"] = target_failed_test_cases
    if target_test_case_pass_rate is not None:
        payload["target_test_case_pass_rate"] = target_test_case_pass_rate
    if target_passed_test_case_ids is not None:
        payload["target_passed_test_case_ids"] = target_passed_test_case_ids
    if target_failed_test_case_ids is not None:
        payload["target_failed_test_case_ids"] = target_failed_test_case_ids
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class CheckAction(Action):
    target_id: str = Field(
        description="Current target id. Format: {repo}::{test_file_path}.",
    )


class CheckObservation(Observation):
    ok: bool
    message: str
    target_id: str
    record_id: str | None = Field(default=None)
    depth: int
    f2p: list[str] = Field(default_factory=list)
    p2p: list[str] = Field(default_factory=list)
    target_test_file_path: str | None = Field(default=None)
    target_file_failed: bool = Field(default=False)
    other_failed_test_files: list[str] = Field(default_factory=list)
    other_failed_test_file_count: int = Field(default=0)
    target_total_test_cases: int | None = Field(default=None)
    target_passed_test_cases: int | None = Field(default=None)
    target_failed_test_cases: int | None = Field(default=None)
    target_test_case_pass_rate: float | None = Field(default=None)
    target_passed_test_case_ids: list[str] = Field(default_factory=list)
    target_failed_test_case_ids: list[str] = Field(default_factory=list)
    workspace_fingerprint: str | None = Field(default=None)

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        summary = [
            f"check_ok: {str(self.ok).lower()}",
            f"message: {self.message}",
            f"target_id: {self.target_id}",
        ]
        if self.record_id:
            summary.append(f"record_id: {self.record_id}")
        summary.extend(
            [
            f"depth: {self.depth}",
            (
                "f2p_count: "
                f"{len(self.f2p)} (baseline-pass test files that now fail)"
            ),
            (
                "p2p_count: "
                f"{len(self.p2p)} (baseline-pass test files that still pass)"
            ),
            (
                "target_file_failed: "
                f"{self.target_file_failed} (whether the current target file is now in f2p)"
            ),
            (
                "other_failed_test_file_count: "
                f"{self.other_failed_test_file_count} "
                "(failed baseline-pass test files other than the current target)"
            ),
            ]
        )
        if self.target_test_file_path:
            summary.append(f"target_test_file_path: {self.target_test_file_path}")
        if self.target_total_test_cases is not None:
            summary.extend(
                [
                    f"target_total_test_cases: {self.target_total_test_cases}",
                    f"target_passed_test_cases: {self.target_passed_test_cases}",
                    f"target_failed_test_cases: {self.target_failed_test_cases}",
                    f"target_test_case_pass_rate: {self.target_test_case_pass_rate}",
                ]
            )
        if self.target_passed_test_case_ids:
            summary.append(
                "target_passed_test_case_ids:\n"
                + "\n".join(self.target_passed_test_case_ids)
            )
        if self.target_failed_test_case_ids:
            summary.append(
                "target_failed_test_case_ids:\n"
                + "\n".join(self.target_failed_test_case_ids)
            )
        if self.other_failed_test_files:
            summary.append(
                "other_failed_test_files:\n"
                + "\n".join(self.other_failed_test_files)
            )
        if self.workspace_fingerprint:
            summary.append(f"workspace_fingerprint: {self.workspace_fingerprint}")
        return [TextContent(text="\n".join(summary))]


class CheckExecutor(ToolExecutor[CheckAction, CheckObservation]):
    def __call__(self, action: CheckAction, conversation=None) -> CheckObservation:  # noqa: ARG002
        try:
            parsed_repo, test_file_path = parse_target_id(action.target_id)
        except ValueError as exc:
            return CheckObservation(
                ok=False,
                message=f"invalid target_id: {exc}",
                target_id=action.target_id,
                depth=0,
            )
        repo = parsed_repo

        try:
            baseline_pass = _load_baseline_pass()
        except ValueError as exc:
            return CheckObservation(
                ok=False,
                message=f"invalid baseline state: {exc}",
                target_id=action.target_id,
                depth=0,
            )

        if test_file_path not in set(baseline_pass):
            return CheckObservation(
                ok=False,
                message=(
                    "target test_file_path is not in baseline_pass; "
                    "current target_id is invalid for depth loop"
                ),
                target_id=action.target_id,
                depth=0,
            )

        try:
            current_depth = _candidate_depth(action.target_id)
        except ValueError as exc:
            return CheckObservation(
                ok=False,
                message=f"invalid depth state: {exc}",
                target_id=action.target_id,
                depth=0,
            )
        current_record_id = format_record_id(action.target_id, current_depth)
        run_tests_path, check_depth_dir = _runtime_paths(action.target_id, current_depth)
        if not run_tests_path.exists():
            return CheckObservation(
                ok=False,
                message=(
                    f"Missing run_tests.py at {run_tests_path}. "
                    "Ensure baseline(depth=0) has been prepared for this repo."
                ),
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        check_depth_dir.mkdir(parents=True, exist_ok=True)
        round_index = _next_check_index(check_depth_dir)
        turn_dir = check_depth_dir / f"turn_{round_index}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        result_path = turn_dir / "result.txt"
        details_path = _details_path(turn_dir)
        meta_path = turn_dir / "meta.json"

        cmd_with_details = [
            "python3",
            str(run_tests_path),
            "--input",
            "/testbed",
            "--output",
            str(result_path),
            "--details-output",
            str(details_path),
            "--details-target-file",
            test_file_path,
        ]
        try:
            timeout_seconds = _resolve_timeout_seconds()
        except ValueError as exc:
            return CheckObservation(
                ok=False,
                message=f"invalid timeout config: {exc}",
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        try:
            proc = subprocess.run(
                cmd_with_details,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CheckObservation(
                ok=False,
                message=f"check timeout after {timeout_seconds}s",
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        if proc.returncode != 0:
            detail_excerpt = ((proc.stderr or proc.stdout or "").strip())[:1200]
            return CheckObservation(
                ok=False,
                message=(
                    f"run_tests.py failed with returncode={proc.returncode}. "
                    f"detail={detail_excerpt}"
                ),
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        if not result_path.exists():
            return CheckObservation(
                ok=False,
                message=f"result file missing: {result_path}",
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        try:
            passed_paths, failed_paths = _parse_result_text_at_least_one_header(
                result_path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            return CheckObservation(
                ok=False,
                message=f"invalid result format: {exc}",
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        baseline_set = set(baseline_pass)
        f2p = sorted(baseline_set & set(failed_paths))
        p2p = sorted(baseline_set & set(passed_paths))
        target_file_failed = test_file_path in f2p
        other_failed_test_files = sorted(
            path for path in f2p if path != test_file_path
        )

        target_details: dict[str, object] | None = None
        try:
            target_details = _load_target_details(
                details_path,
                expected_target_test_file_path=test_file_path,
            )
        except ValueError as exc:
            return CheckObservation(
                ok=False,
                message=f"invalid details format: {exc}",
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        try:
            workspace_fingerprint = _compute_workspace_fingerprint()
        except RuntimeError as exc:
            return CheckObservation(
                ok=False,
                message=f"failed to compute workspace fingerprint: {exc}",
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        try:
            _write_check_meta(
                meta_path=meta_path,
                target_id=action.target_id,
                record_id=current_record_id,
                repo=repo,
                depth=current_depth,
                round_index=round_index,
                result_path=result_path,
                workspace_fingerprint=workspace_fingerprint,
                f2p=f2p,
                p2p=p2p,
                target_test_file_path=test_file_path,
                target_file_failed=target_file_failed,
                other_failed_test_files=other_failed_test_files,
                details_path=details_path if target_details is not None else None,
                target_total_test_cases=(
                    target_details.get("target_total_test_cases")
                    if target_details is not None
                    else None
                ),
                target_passed_test_cases=(
                    target_details.get("target_passed_test_cases")
                    if target_details is not None
                    else None
                ),
                target_failed_test_cases=(
                    target_details.get("target_failed_test_cases")
                    if target_details is not None
                    else None
                ),
                target_test_case_pass_rate=(
                    target_details.get("target_test_case_pass_rate")
                    if target_details is not None
                    else None
                ),
                target_passed_test_case_ids=(
                    target_details.get("target_passed_test_case_ids")
                    if target_details is not None
                    else None
                ),
                target_failed_test_case_ids=(
                    target_details.get("target_failed_test_case_ids")
                    if target_details is not None
                    else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return CheckObservation(
                ok=False,
                message=f"failed to write check metadata: {exc}",
                target_id=action.target_id,
                record_id=current_record_id,
                depth=current_depth,
            )

        return CheckObservation(
            ok=True,
            message=(
                f"check ok: depth={current_depth}, round={round_index}, "
                f"result={result_path}, meta={meta_path}"
            ),
            target_id=action.target_id,
            record_id=current_record_id,
            depth=current_depth,
            f2p=f2p,
            p2p=p2p,
            target_test_file_path=test_file_path,
            target_file_failed=target_file_failed,
            other_failed_test_files=other_failed_test_files,
            other_failed_test_file_count=len(other_failed_test_files),
            target_total_test_cases=(
                int(target_details["target_total_test_cases"])
                if target_details is not None
                else None
            ),
            target_passed_test_cases=(
                int(target_details["target_passed_test_cases"])
                if target_details is not None
                else None
            ),
            target_failed_test_cases=(
                int(target_details["target_failed_test_cases"])
                if target_details is not None
                else None
            ),
            target_test_case_pass_rate=(
                float(target_details["target_test_case_pass_rate"])
                if target_details is not None
                else None
            ),
            target_passed_test_case_ids=(
                list(target_details["target_passed_test_case_ids"])
                if target_details is not None
                else []
            ),
            target_failed_test_case_ids=(
                list(target_details["target_failed_test_case_ids"])
                if target_details is not None
                else []
            ),
            workspace_fingerprint=workspace_fingerprint,
        )


class CheckTool(ToolDefinition[CheckAction, CheckObservation]):
    """Tool that checks current task-maker depth status."""

    @classmethod
    def create(cls, conv_state, **kwargs):  # noqa: ARG003
        return [
            cls(
                description=(
                    "Validate the current target at the current depth. "
                    "Input requires target_id only. "
                    "The tool reads current_depth from shared runtime state, executes run_tests.py, validates both file-level output and target-file details output, computes f2p/p2p against baseline_pass, and writes per-round artifacts under /output/records/<target_id_slug>/check/depth_<k>/turn_<n>/. "
                    "Each successful check corresponds to record_id = {target_id}::depth=<current_depth>. "
                    "Do not call it repeatedly without meaningful code changes."
                ),
                action_type=CheckAction,
                observation_type=CheckObservation,
                executor=CheckExecutor(),
            )
        ]


def register_check_tool() -> None:
    register_tool("check", CheckTool.create)


# Ensure tool is registered when module is imported on the client side
register_check_tool()
