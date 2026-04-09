"""Validator tool for checking agent outputs."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import Field

from openhands.sdk import ImageContent, TextContent
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

_VALIDATOR_CALL_COUNT = 0
_VALIDATOR_CALL_LOCK = threading.Lock()
_CLEANUP_REQUEST_TIMEOUT_SECONDS = 60


@dataclass
class ValidationResult:
    ok: bool
    message: str
    dockerfile_text: str | None = None
    used_image_name: str | None = None
    image_history: list[str] | None = None
    passed_num: int | None = None


class ValidatorAction(Action):
    dockerfile_path: str = Field(
        description="Workspace path to the current Dockerfile under validation.",
    )
    test_script_path: str = Field(
        description="Workspace path to the current run_tests.py under validation.",
    )
    extra_info_path: str = Field(
        description="Workspace path to the current extra_info.json for validation status updates.",
    )
    image_name: str = Field(
        description=(
            "Validation image selector. Use 'scratch' to build a new image, or provide an existing image name to reuse."
        ),
    )


class ValidatorObservation(Observation):
    ok: bool
    message: str

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        summary = [
            f"validator_ok: {str(self.ok).lower()}",
            f"message: {self.message}",
        ]
        return [TextContent(text="\n".join(summary))]


class ValidatorExecutor(ToolExecutor[ValidatorAction, ValidatorObservation]):
    def __init__(self) -> None:
        self._tracked_image_names: set[str] = set()
        self._tracked_lock = threading.Lock()

    def _track_image_names(self, image_names: list[str]) -> None:
        if not image_names:
            return
        with self._tracked_lock:
            for image_name in image_names:
                normalized = image_name.strip()
                if normalized:
                    self._tracked_image_names.add(normalized)

    def _flush_tracked_images(self, host_task_dir: str) -> None:
        with self._tracked_lock:
            image_names = sorted(self._tracked_image_names)
        if not image_names:
            return

        payload = {
            "image_names": image_names,
            "host_task_dir": host_task_dir,
        }
        host_gateway_ip = os.getenv("HOST_GATEWAY_IP", "172.17.0.1")
        port = int(os.getenv("VALIDATOR_PORT", "9090"))
        url = f"http://{host_gateway_ip}:{port}/cleanup_images"

        try:
            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=_CLEANUP_REQUEST_TIMEOUT_SECONDS):
                pass
            with self._tracked_lock:
                self._tracked_image_names.difference_update(image_names)
        except Exception:  # noqa: BLE001
            return

    @staticmethod
    def _build_observation(
        *,
        ok: bool,
        message: str,
        used_image_name: str | None,
    ) -> ValidatorObservation:
        used_name = used_image_name.strip() if isinstance(used_image_name, str) else ""
        if not used_name:
            used_name = "N/A"
        suffix = f"used_image_name: {used_name}"
        return ValidatorObservation(
            ok=ok,
            message=suffix if not message.rstrip() else f"{message.rstrip()}\n{suffix}",
        )

    def __call__(
        self,
        action: ValidatorAction,
        conversation=None,
    ) -> ValidatorObservation:  # noqa: ARG002
        image_name = action.image_name.strip()

        # Check /testbed state
        testbed_error = _check_testbed_unchanged()
        if testbed_error is not None:
            _update_extra_info(action.extra_info_path, status="failed")
            return self._build_observation(
                ok=False,
                message=testbed_error,
                used_image_name=None,
            )

        # Detect test script legality
        legal_check = None
        legal_reason = None
        parse_failure_message = None
        if conversation is not None and hasattr(conversation, "ask_agent"):
            legal_check, legal_reason, parse_failure_message = _detect_test_script_legal(
                conversation,
                action.test_script_path,
            )
        if legal_check is not None:
            _update_extra_info(
                action.extra_info_path,
                legal=legal_check,
                reason=legal_reason,
            )
        if legal_check is False:
            reason = f" Reason: {legal_reason}" if legal_reason else ""
            _update_extra_info(action.extra_info_path, status="failed")
            return self._build_observation(
                ok=False,
                message="Illegal test script detected.\n"
                f"{reason}",
                used_image_name=None,
            )
        if parse_failure_message is not None:
            _update_extra_info(action.extra_info_path, status="failed")
            return self._build_observation(
                ok=False,
                message=parse_failure_message,
                used_image_name=None,
            )

        # Request host validation
        try:
            host_task_dir = _read_host_task_dir()
        except RuntimeError as exc:
            _update_extra_info(action.extra_info_path, status="failed")
            return self._build_observation(
                ok=False,
                message=str(exc),
                used_image_name=None,
            )
        call_count = _next_validator_call_count()
        result = request_host_validation(
            dockerfile_path=action.dockerfile_path,
            test_script_path=action.test_script_path,
            extra_info_path=action.extra_info_path,
            host_task_dir=host_task_dir,
            call_count=call_count,
            image_name=image_name,
        )
        # Only track images when image_name is scratch and an image name is actually returned (indicating the build succeeded), then delete them in bulk when this task ends.
        cleanup_candidates: list[str] = []
        if image_name.lower() == "scratch" and result.used_image_name:
            cleanup_candidates.append(result.used_image_name)
        if result.image_history:
            cleanup_candidates.extend(result.image_history)
        self._track_image_names(cleanup_candidates)
        _update_extra_info(
            action.extra_info_path,
            status="success" if result.ok else "failed",
            passed_num=result.passed_num if result.ok else None,
        )
        if result.ok and conversation is not None:
            if result.dockerfile_text:
                try:
                    Path(action.dockerfile_path).write_text(result.dockerfile_text)
                except Exception as exc:  # noqa: BLE001
                    _update_extra_info(action.extra_info_path, status="failed")
                    return self._build_observation(
                        ok=False,
                        message=f"Validation succeeded but failed to update Dockerfile: {exc}",
                        used_image_name=result.used_image_name,
                    )
            conversation.state.execution_status = ConversationExecutionStatus.FINISHED
        return self._build_observation(
            ok=result.ok,
            message=result.message,
            used_image_name=result.used_image_name,
        )

    def close(self) -> None:
        try:
            host_task_dir = _read_host_task_dir()
        except RuntimeError:
            return
        self._flush_tracked_images(host_task_dir)


class ValidatorTool(ToolDefinition[ValidatorAction, ValidatorObservation]):
    """Tool that validates builder artifacts by building and running tests."""

    @classmethod
    def create(cls, conv_state, **kwargs):  # noqa: ARG003
        return [
            cls(
                description=(
                    "Run host-side validation for the current Dockerfile, run_tests.py, and extra_info.json. "
                    "Call this only after local validation is ready. "
                    "This tool triggers expensive host-side image build and test execution. "
                    "You must provide image_name: use 'scratch' to build a new image, or provide an existing image name to reuse."
                ),
                action_type=ValidatorAction,
                observation_type=ValidatorObservation,
                executor=ValidatorExecutor(),
            )
        ]


def _update_extra_info(
    extra_info_path: str,
    *,
    status: str | None = None,
    legal: bool | None = None,
    reason: str | None = None,
    passed_num: int | None = None,
) -> None:
    path = Path(extra_info_path)
    payload: dict[str, Any]
    try:
        payload = json.loads(path.read_text()) if path.exists() else {}
    except Exception:  # noqa: BLE001
        payload = {}
    if status is not None:
        payload["status"] = status
    if legal is not None:
        payload["legal"] = bool(legal)
    if reason is not None:
        payload["reason"] = reason
    if passed_num is not None:
        payload["passed_num"] = int(passed_num)
    timeout_env = os.getenv("RUN_TESTS_TIMEOUT")
    if timeout_env:
        try:
            payload["run_tests_timeout"] = int(timeout_env)
        except ValueError:
            pass
    path.write_text(json.dumps(payload, indent=2))

def _check_testbed_unchanged() -> str | None:
    base_commit_path = Path("/store/testbed_base_commit")
    try:
        baseline_commit = base_commit_path.read_text().strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Unable to verify /testbed state: baseline commit is missing. Please reinitialize the workspace."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Unable to verify /testbed state: baseline commit is missing. Please reinitialize the workspace."
        ) from exc
    if not baseline_commit:
        raise RuntimeError(
            "Unable to verify /testbed state: baseline commit is missing. Please reinitialize the workspace."
        )
    
    try:
        repo_check = _run_git(["rev-parse", "--is-inside-work-tree"])
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Unable to verify /testbed state: git not found ({exc})."
        ) from exc
    if repo_check.returncode != 0:
        raise RuntimeError(
            f"Unable to verify /testbed state: not a git repository. {repo_check.stderr.strip()}"
        )
    
    head = _run_git(["rev-parse", "HEAD"])
    if head.returncode != 0:
        raise RuntimeError(
            f"Unable to verify /testbed state: failed to read HEAD. {head.stderr.strip()}"
        )
    status = _run_git(["status", "--porcelain"])
    if status.returncode != 0:
        raise RuntimeError(
            f"Unable to verify /testbed state: failed to read status. {status.stderr.strip()}"
        )
    
    head_value = head.stdout.strip()
    status_value = status.stdout.strip()
    if head_value != baseline_commit or status_value:
        _restore_testbed(baseline_commit)
        details = status_value or "(empty)"
        message_lines = [
            "We detected changes under /testbed and restored them.",
            "git status --porcelain:",
            details,
        ]
        if head_value != baseline_commit and not status_value:
            message_lines.append(
                f"Note: current HEAD={head_value} differs from baseline {baseline_commit}."
            )
        message_lines.extend(
            [
                "If the entries above were not intentionally modified or added by you, or you believe they do not affect the generated Dockerfile/run_tests.py, you may try calling the validator tool again.",
                "If they were intentionally modified or added, please review your artifacts and ensure run_tests.py runs successfully locally before retrying.",
            ]
        )
        return "\n".join(message_lines)
    return None


def _restore_testbed(baseline_commit: str) -> None:
    _run_git(["checkout", "--detach", baseline_commit])
    _run_git(["reset", "--hard", baseline_commit])
    _run_git(["clean", "-fd"])
    _run_git(["submodule", "update", "--init", "--recursive", "--depth", "1"])


def _run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(Path("/testbed")), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _detect_test_script_legal(
    conversation, test_script_path: str
) -> tuple[bool | None, str | None, str | None]:
    try:
        test_script_text = Path(test_script_path).read_text()
    except Exception:  # noqa: BLE001
        return None, None, None

    base_prompt = "You are reviewing run_tests.py. Decide if it is legal.\n" \
        "Rules:\n" \
        "1) It must actually execute the repo's real unit tests (via a real test framework like pytest/unittest, OR direct invocation of real test_* functions) and derive passed/failed lists from actual outcomes.\n" \
        "2) Any failing test must be recorded as failed (or cause overall failure). Swallowing failures is cheating. Examples of cheating:\n" \
        "   - Catching exceptions from test_* calls without re-raising AND without recording failure.\n" \
        "   - Unconditionally appending a test file to passed even if a test failed.\n" \
        "   - Hardcoding or fabricating passed/failed lists.\n" \
        "3) It must support --input/--output and use them to read inputs and write outputs.\n\n" \
        f"Path: {test_script_path}\n" \
        f"Script:\n{test_script_text}\n\n" \
        "Return JSON only: {\"legal\": true/false, \"reason\": \"...\"}."
    
    had_error = False
    last_error: str | None = None
    last_response: str | None = None
    for prompt in (base_prompt, base_prompt, base_prompt):
        try:
            response = conversation.ask_agent(prompt)
        except Exception as exc:  # noqa: BLE001
            had_error = True
            last_error = str(exc)
            continue
        last_response = response
        parsed = _parse_legal_response(response)
        if parsed is not None:
            return parsed[0], parsed[1], None
    # Calling ask_agent() failed, returning a generic response
    if had_error:
        return (
            None,
            None,
            "please ensure the passed test files and failed test files in your test script are generated from actual test execution. "
            f"ask_agent error: {last_error}",
        )
    if last_response is not None:
        preview = last_response.strip() or "(empty response)"
        return (
            None,
            None,
            "Failed to parse ask_agent response. Please ensure the passed test files and failed test files in your test script are generated from actual test execution. "
            f"ask_agent response: {preview}",
        )
    # Analysis failed. Returning a generic response.
    return (
        None,
        None,
        "Please ensure the passed test files and failed test files in your test script are generated from actual test execution.",
    )


def _parse_legal_response(response: str) -> tuple[bool, str | None] | None:
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
    legal_value = payload.get("legal")
    reason_value = payload.get("reason")
    reason_text = reason_value if isinstance(reason_value, str) else None
    if isinstance(legal_value, bool):
        return legal_value, reason_text
    if isinstance(legal_value, str):
        normalized = legal_value.strip().lower()
        if normalized in {"true", "yes", "1", "correct", "legal", "true."}:
            return True, reason_text
        if normalized in {"false", "no", "0", "incorrect", "illegal", "false."}:
            return False, reason_text
    return None


def request_host_validation(
    *,
    dockerfile_path: str,
    test_script_path: str,
    extra_info_path: str,
    host_task_dir: str,
    call_count: int | None = None,
    image_name: str,
) -> ValidationResult:
    try:
        dockerfile_text = Path(dockerfile_path).read_text()
        test_script_text = Path(test_script_path).read_text()
        extra_info_text = Path(extra_info_path).read_text()
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, f"Failed to read artifacts: {exc}")

    payload = {
        "dockerfile": dockerfile_text,
        "test_script": test_script_text,
        "extra_info": extra_info_text,
        "host_task_dir": host_task_dir,
        "call_count": call_count or 0,
        "image_name": image_name,
    }

    host_gateway_ip = os.getenv("HOST_GATEWAY_IP", "172.17.0.1")
    port = int(os.getenv("VALIDATOR_PORT", "9090"))
    url = f"http://{host_gateway_ip}:{port}/validate"

    try:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        return ValidationResult(False, f"Host validation request failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, f"Invalid response from host: {exc}")

    return ValidationResult(
        bool(data.get("ok")),
        str(data.get("message", "")),
        data.get("dockerfile") if isinstance(data.get("dockerfile"), str) else None,
        data.get("used_image_name")
        if isinstance(data.get("used_image_name"), str)
        else None,
        [
            value
            for value in data.get("image_history", [])
            if isinstance(value, str)
        ]
        if isinstance(data.get("image_history"), list)
        else None,
        data.get("passed_num") if isinstance(data.get("passed_num"), int) else None,
    )


def register_validator_tool() -> None:
    register_tool("validator", ValidatorTool.create)


def _read_host_task_dir() -> str:
    path = Path("/store/host_task_dir")
    try:
        value = path.read_text().strip()
    except FileNotFoundError:
        raise RuntimeError("Missing required /store/host_task_dir; cannot call validator.")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read /store/host_task_dir: {exc}") from exc
    if not value:
        raise RuntimeError("Empty /store/host_task_dir; cannot call validator.")
    return value


def _next_validator_call_count() -> int:
    global _VALIDATOR_CALL_COUNT
    with _VALIDATOR_CALL_LOCK:
        _VALIDATOR_CALL_COUNT += 1
        return _VALIDATOR_CALL_COUNT


# Ensure tool is registered when module is imported on the client side
register_validator_tool()
