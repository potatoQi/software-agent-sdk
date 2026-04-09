"""Save tool for task-maker target/depth snapshots."""

from __future__ import annotations

from collections.abc import Sequence
import json
import subprocess
from hashlib import sha1
from pathlib import Path

from pydantic import BaseModel, Field

from alpha.shared.record_id_codec import format_record_id
from alpha.shared.target_test_details_schema import validate_target_details_payload
from openhands.sdk import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from alpha.shared.target_id_codec import parse_target_id, target_id_to_slug


def _baseline_json_path() -> Path:
    return Path("/output/baseline/baseline.json")


def _load_baseline_json() -> dict:
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
    commit0_raw = payload.get("commit0")
    if commit0_raw is None:
        payload["commit0"] = ""
    elif not isinstance(commit0_raw, str):
        raise ValueError("invalid baseline json: commit0 must be a string")
    payload["baseline_pass"] = sorted(set(baseline_pass))
    return payload


def _check_dir(target_id: str, depth: int) -> Path:
    target_id_slug = target_id_to_slug(target_id)
    return Path(f"/output/records/{target_id_slug}/check/depth_{depth}")


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


def _advance_depth_state(target_id: str, next_depth: int) -> None:
    if next_depth < 1:
        raise ValueError("next_depth must be >= 1")
    state_path = _depth_state_path(target_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "target_id": target_id,
                "current_depth": next_depth,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_check_meta(meta_path: Path) -> dict:
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"invalid check meta {meta_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid check meta {meta_path}: root must be an object")
    return payload


def _find_latest_success_meta(*, check_dir: Path, target_id: str, depth: int) -> dict | None:
    if not check_dir.exists():
        return None

    expected_record_id = format_record_id(target_id, depth)
    candidates: list[dict] = []
    meta_paths = sorted(check_dir.glob("turn_*/meta.json"))
    if not meta_paths:
        meta_paths = sorted(check_dir.glob("check_*.meta.json"))
    for meta_path in meta_paths:
        payload = _load_check_meta(meta_path)
        if payload.get("ok") is not True:
            continue
        payload_target_id = payload.get("target_id", payload.get("record_id"))
        if payload_target_id != target_id:
            continue
        payload_record_id = payload.get("record_id")
        if payload_record_id is not None and payload_record_id != expected_record_id:
            raise ValueError(
                f"invalid check meta {meta_path}: record_id must equal current target_id+depth"
            )
        if payload.get("depth") != depth:
            continue
        round_raw = payload.get("round")
        if not isinstance(round_raw, int) or round_raw <= 0:
            raise ValueError(
                f"invalid check meta {meta_path}: round must be a positive integer"
            )
        workspace_fingerprint = payload.get("workspace_fingerprint")
        if not isinstance(workspace_fingerprint, str) or not workspace_fingerprint:
            raise ValueError(
                f"invalid check meta {meta_path}: workspace_fingerprint must be a non-empty string"
            )
        f2p_raw = payload.get("f2p")
        if not isinstance(f2p_raw, list) or any(
            not isinstance(item, str) for item in f2p_raw
        ):
            raise ValueError(f"invalid check meta {meta_path}: f2p must be list[str]")
        p2p_raw = payload.get("p2p")
        if not isinstance(p2p_raw, list) or any(
            not isinstance(item, str) for item in p2p_raw
        ):
            raise ValueError(f"invalid check meta {meta_path}: p2p must be list[str]")
        target_test_file_path = payload.get("target_test_file_path")
        if (
            not isinstance(target_test_file_path, str)
            or not target_test_file_path.startswith("/testbed/")
        ):
            raise ValueError(
                f"invalid check meta {meta_path}: target_test_file_path must start with /testbed/"
            )
        payload.update(
            validate_target_details_payload(
                payload,
                expected_target_test_file_path=target_test_file_path,
                error_prefix=f"invalid check meta {meta_path}",
            )
        )
        payload["workspace_fingerprint"] = workspace_fingerprint
        payload["f2p"] = sorted(set(f2p_raw))
        payload["p2p"] = sorted(set(p2p_raw))
        candidates.append(payload)

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["round"])


def _git_run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"git command failed: {' '.join(args)}: {exc}") from exc


def _compute_workspace_fingerprint() -> str:
    try:
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
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"fingerprint command failed: {exc}") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"fingerprint command failed with returncode={proc.returncode}: {stderr}"
        )
    return sha1(proc.stdout or b"").hexdigest()


def _workspace_is_dirty() -> bool:
    proc = _git_run(
        [
            "-C",
            "/testbed",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git status failed: {detail}")
    return bool((proc.stdout or "").strip())


def _git_identity_effective() -> tuple[bool, str]:
    author = _git_run(["-C", "/testbed", "var", "GIT_AUTHOR_IDENT"])
    if author.returncode != 0:
        detail = (author.stderr or author.stdout or "").strip()
        return False, f"GIT_AUTHOR_IDENT unavailable: {detail}"
    committer = _git_run(["-C", "/testbed", "var", "GIT_COMMITTER_IDENT"])
    if committer.returncode != 0:
        detail = (committer.stderr or committer.stdout or "").strip()
        return False, f"GIT_COMMITTER_IDENT unavailable: {detail}"
    return True, ""


def _save_output_path(target_id: str, depth: int) -> Path:
    target_id_slug = target_id_to_slug(target_id)
    return Path(f"/output/records/{target_id_slug}/save/depth_{depth}/save.json")


def _write_save_output(task_record: "SaveTaskRecord") -> Path:
    output_path = _save_output_path(task_record.target_id, task_record.depth)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = task_record.model_dump()
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _build_issue_hint_prompt(
    *,
    gold_patch: str,
) -> str:
    return (
        "You are labeling one saved feature-breaking diff.\n"
        "Use the current conversation context and the diff below.\n"
        'Return JSON only: {"issue": "...", "hint": "..."}\n\n'
        "Rules:\n"
        "1) issue: a concise natural-language description of the broken functionality that should be restored. If multiple related behaviors or feature areas are broken, describe them together briefly.\n"
        "2) Focus on behavior and functionality, not tests, file names, patch operations, or line-level edits.\n"
        "3) hint: a concise high-level restoration direction; do not reveal the exact fix.\n"
        "4) Both values must be non-empty plain strings. No markdown fences and no extra text.\n\n"
        "diff(commit0, commit_k):\n"
        f"{gold_patch}"
    )


def _parse_issue_hint_response(response: str) -> tuple[str, str] | None:
    response = response.strip()
    if not response:
        return None

    payload = None
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                payload = json.loads(response[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(payload, dict):
        return None
    issue = payload.get("issue")
    hint = payload.get("hint")
    if not isinstance(issue, str) or not issue.strip():
        return None
    if not isinstance(hint, str) or not hint.strip():
        return None
    return issue.strip(), hint.strip()


def _generate_issue_hint_from_diff(
    *,
    conversation,
    gold_patch: str,
) -> tuple[str, str]:
    if conversation is None or not hasattr(conversation, "ask_agent"):
        raise RuntimeError("conversation.ask_agent is unavailable")

    prompt = _build_issue_hint_prompt(
        gold_patch=gold_patch,
    )

    had_error = False
    last_error: str | None = None
    last_response: str | None = None
    for _ in range(3):
        try:
            response = conversation.ask_agent(prompt)
        except Exception as exc:  # noqa: BLE001
            had_error = True
            last_error = str(exc)
            continue
        last_response = response
        parsed = _parse_issue_hint_response(response)
        if parsed is not None:
            return parsed

    if had_error:
        raise RuntimeError(f"ask_agent error: {last_error}")
    if last_response is not None:
        preview = last_response.strip() or "(empty response)"
        raise RuntimeError(f"failed to parse ask_agent response: {preview}")
    raise RuntimeError("ask_agent returned no usable response")


class SaveAction(Action):
    target_id: str = Field(
        description="Current target id. Format: {repo}::{test_file_path}.",
    )


class SaveTaskRecord(BaseModel):
    target_id: str
    record_id: str
    repo: str
    commit: str
    f2p: list[str] = Field(default_factory=list)
    p2p: list[str] = Field(default_factory=list)
    gold_patch: str
    issue: str
    depth: int
    hint: str
    target_total_test_cases: int
    target_passed_test_cases: int
    target_failed_test_cases: int
    target_test_case_pass_rate: float
    target_passed_test_case_ids: list[str] = Field(default_factory=list)
    target_failed_test_case_ids: list[str] = Field(default_factory=list)


class SaveObservation(Observation):
    ok: bool
    message: str
    task_record: SaveTaskRecord | None = None

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        summary = [
            f"save_ok: {str(self.ok).lower()}",
            f"message: {self.message}",
        ]
        if self.task_record is not None:
            summary.append(f"target_id: {self.task_record.target_id}")
            summary.append(f"record_id: {self.task_record.record_id}")
            summary.append(f"depth: {self.task_record.depth}")
            summary.append(f"commit: {self.task_record.commit}")
            summary.append(
                f"target_total_test_cases: {self.task_record.target_total_test_cases}"
            )
            summary.append(
                f"target_failed_test_cases: {self.task_record.target_failed_test_cases}"
            )
        return [TextContent(text="\n".join(summary))]


class SaveExecutor(ToolExecutor[SaveAction, SaveObservation]):
    @staticmethod
    def _error(message: str) -> SaveObservation:
        return SaveObservation(ok=False, message=message, task_record=None)

    @classmethod
    def _save_failed(cls, detail: str) -> SaveObservation:
        return cls._error(f"save_failed: {detail}")

    @classmethod
    def _post_commit_failure(cls, commit_k: str, detail: str) -> SaveObservation:
        return cls._error(
            "post_commit_failure: repo already mutated after commit; "
            f"commit_k={commit_k}; detail={detail}"
        )

    @staticmethod
    def _proc_detail(proc: subprocess.CompletedProcess[str]) -> str:
        return (proc.stderr or proc.stdout or "").strip()

    @classmethod
    def _reset_index_after_failure(cls) -> tuple[bool, str]:
        try:
            proc = _git_run(["-C", "/testbed", "reset"])
        except RuntimeError as exc:
            return False, str(exc)
        if proc.returncode != 0:
            return False, cls._proc_detail(proc)
        return True, ""

    @classmethod
    def _save_failed_with_reset(cls, detail: str) -> SaveObservation:
        reset_ok, reset_detail = cls._reset_index_after_failure()
        if reset_ok:
            return cls._save_failed(detail)
        return cls._save_failed(
            f"{detail}; rollback_failed: {reset_detail}; index may be left staged"
        )

    def _run_precheck(
        self, action: SaveAction
    ) -> tuple[str, dict, int] | SaveObservation:
        try:
            parsed_repo, test_file_path = parse_target_id(action.target_id)
        except ValueError as exc:
            return self._save_failed(f"invalid target_id: {exc}")
        repo = parsed_repo

        try:
            baseline = _load_baseline_json()
        except ValueError as exc:
            return self._save_failed(f"invalid baseline state: {exc}")

        commit0 = str(baseline.get("commit0", "")).strip()
        baseline_pass = set(baseline["baseline_pass"])
        if not commit0:
            return self._save_failed("missing commit0 in baseline.json")
        if test_file_path not in baseline_pass:
            return self._save_failed("target test_file_path is not in baseline_pass")

        try:
            current_depth = _candidate_depth(action.target_id)
        except ValueError as exc:
            return self._save_failed(f"invalid depth state: {exc}")
        check_dir = _check_dir(action.target_id, current_depth)
        try:
            meta = _find_latest_success_meta(
                check_dir=check_dir,
                target_id=action.target_id,
                depth=current_depth,
            )
        except ValueError as exc:
            return self._save_failed(f"invalid check meta state: {exc}")

        if meta is None:
            return self._save_failed("no matching successful check metadata found")

        workspace_fingerprint = meta["workspace_fingerprint"]
        try:
            current_fingerprint = _compute_workspace_fingerprint()
        except RuntimeError as exc:
            return self._save_failed(f"failed to compute workspace fingerprint: {exc}")

        if current_fingerprint != workspace_fingerprint:
            return self._save_failed(
                "workspace changed after last successful check; rerun check_tool first"
            )

        try:
            is_dirty = _workspace_is_dirty()
        except RuntimeError as exc:
            return self._save_failed(str(exc))

        if not is_dirty:
            return self._save_failed("workspace clean; nothing to save")

        try:
            identity_ok, identity_msg = _git_identity_effective()
        except RuntimeError as exc:
            return self._save_failed(f"git identity check failed: {exc}")

        if not identity_ok:
            return self._save_failed(f"git identity missing: {identity_msg}")

        meta["repo"] = repo
        return commit0, meta, current_depth

    def __call__(self, action: SaveAction, conversation=None) -> SaveObservation:  # noqa: ARG002
        precheck = self._run_precheck(action)
        if isinstance(precheck, SaveObservation):
            return precheck
        commit0, meta, current_depth = precheck
        current_record_id = format_record_id(action.target_id, current_depth)

        try:
            add_proc = _git_run(["-C", "/testbed", "add", "-A"])
        except RuntimeError as exc:
            return self._save_failed_with_reset(f"git add failed: {exc}")
        if add_proc.returncode != 0:
            return self._save_failed_with_reset(
                f"git add failed: {self._proc_detail(add_proc)}"
            )

        try:
            commit_proc = _git_run(
                [
                    "-C",
                    "/testbed",
                    "commit",
                    "-m",
                    f"task_maker save depth={current_depth}",
                ]
            )
        except RuntimeError as exc:
            return self._save_failed_with_reset(f"git commit failed: {exc}")
        if commit_proc.returncode != 0:
            return self._save_failed_with_reset(
                f"git commit failed: {self._proc_detail(commit_proc)}"
            )

        commit_k = "unknown"
        try:
            head_proc = _git_run(["-C", "/testbed", "rev-parse", "HEAD"])
        except RuntimeError as exc:
            return self._post_commit_failure(commit_k, f"rev-parse HEAD failed: {exc}")
        if head_proc.returncode != 0:
            return self._post_commit_failure(
                commit_k,
                f"rev-parse HEAD failed: {self._proc_detail(head_proc)}",
            )
        commit_k = (head_proc.stdout or "").strip()
        if not commit_k:
            return self._post_commit_failure(commit_k, "empty HEAD commit")

        try:
            diff_proc = _git_run(["-C", "/testbed", "diff", "--binary", commit0, commit_k])
        except RuntimeError as exc:
            return self._post_commit_failure(commit_k, f"git diff failed: {exc}")
        if diff_proc.returncode != 0:
            return self._post_commit_failure(
                commit_k,
                f"git diff failed: {self._proc_detail(diff_proc)}",
            )
        gold_patch = diff_proc.stdout or ""
        if not gold_patch.strip():
            return self._post_commit_failure(commit_k, "gold_patch is empty")

        try:
            issue, hint = _generate_issue_hint_from_diff(
                conversation=conversation,
                gold_patch=gold_patch,
            )
        except RuntimeError as exc:
            return self._post_commit_failure(
                commit_k,
                f"issue/hint generation failed: {exc}",
            )

        task_record = SaveTaskRecord(
            target_id=action.target_id,
            record_id=current_record_id,
            repo=meta["repo"],
            commit=commit_k,
            f2p=meta["f2p"],
            p2p=meta["p2p"],
            gold_patch=gold_patch,
            issue=issue,
            depth=current_depth,
            hint=hint,
            target_total_test_cases=meta["target_total_test_cases"],
            target_passed_test_cases=meta["target_passed_test_cases"],
            target_failed_test_cases=meta["target_failed_test_cases"],
            target_test_case_pass_rate=meta["target_test_case_pass_rate"],
            target_passed_test_case_ids=meta["target_passed_test_case_ids"],
            target_failed_test_case_ids=meta["target_failed_test_case_ids"],
        )
        try:
            _write_save_output(task_record)
        except Exception as exc:  # noqa: BLE001
            return self._post_commit_failure(
                commit_k,
                f"failed to write save output: {exc}",
            )
        try:
            _advance_depth_state(action.target_id, current_depth + 1)
        except Exception as exc:  # noqa: BLE001
            return self._post_commit_failure(
                commit_k,
                f"failed to advance depth state: {exc}",
            )
        return SaveObservation(ok=True, message="save ok", task_record=task_record)


class SaveTool(ToolDefinition[SaveAction, SaveObservation]):
    """Tool that saves one depth snapshot into a task record."""

    @classmethod
    def create(cls, conv_state, **kwargs):  # noqa: ARG003
        return [
            cls(
                description=(
                    "Save the current target/depth snapshot. "
                    "Input requires target_id only. "
                    "The tool reads current_depth from shared runtime state, requires a matching successful check for the same target and depth, and requires the workspace to still match that checked snapshot. "
                    "On success it commits current /testbed changes, builds gold_patch from diff(commit0, commit_k), generates issue/hint from the saved diff, writes /output/records/<target_id_slug>/save/depth_<k>/save.json, and advances to the next depth. "
                    "If a failure happens after commit, the tool returns post_commit_failure."
                ),
                action_type=SaveAction,
                observation_type=SaveObservation,
                executor=SaveExecutor(),
            )
        ]


def register_save_tool() -> None:
    register_tool("save", SaveTool.create)


# Ensure tool is registered when module is imported on the client side
register_save_tool()
