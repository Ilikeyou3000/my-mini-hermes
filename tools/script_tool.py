"""执行可信技能的 Python 脚本；路径限制不是操作系统沙箱。"""

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from skills import manager
from tool_registry import registry

OUTPUT_LIMIT = 32768


def _result(**values):
    return json.dumps(values, ensure_ascii=False)


def _stop(process):
    # 即使主进程已退出，也清理同一进程组内仍持有管道的子进程。
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run(script, args, cwd, timeout):
    output = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = truncated = False
    process = subprocess.Popen(
        [sys.executable, "-u", str(script), *args],
        cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        shell=False, start_new_session=True,
    )
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    budget = OUTPUT_LIMIT - sum(map(len, output.values()))
                    output[key.data].extend(chunk[:budget])
                    if len(chunk) > budget:
                        truncated = True
                        break
                if truncated:
                    break
            if not timed_out and not truncated:
                try:
                    process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
    finally:
        _stop(process)
        process.stdout.close()
        process.stderr.close()
    return _result(
        success=process.returncode == 0 and not timed_out and not truncated,
        exit_code=process.returncode, timed_out=timed_out, truncated=truncated,
        **{name: value.decode("utf-8", errors="replace") for name, value in output.items()},
    )


def skill_run(name: str, file_path: str, args: list[str] = None, timeout: int = 30) -> str:
    """只选取已发现技能的 scripts 内 Python 文件，不接受 shell 命令。"""
    if manager._skill_loader is None or manager._skills_dir is None:
        return _result(success=False, error="skills not initialized")
    if os.name != "posix":
        return _result(success=False, error="script execution requires macOS or Linux")
    if not isinstance(name, str) or not name.strip():
        return _result(success=False, error="name must be a non-empty string")
    if not isinstance(file_path, str) or not file_path or "\x00" in file_path:
        return _result(success=False, error="file_path must be a non-empty path")
    if type(timeout) is not int or not 1 <= timeout <= 120:
        return _result(success=False, error="timeout must be an integer from 1 to 120")
    args = [] if args is None else args
    if (not isinstance(args, list) or len(args) > 100
            or any(not isinstance(arg, str) or "\x00" in arg for arg in args)):
        return _result(success=False, error="args must be an array of at most 100 strings")
    try:
        argument_bytes = sum(len(arg.encode("utf-8")) for arg in args)
    except UnicodeEncodeError:
        return _result(success=False, error="args must contain valid UTF-8 strings")
    if argument_bytes > 32768:
        return _result(success=False, error="arguments exceed 32768 bytes")
    relative = Path(file_path)
    if (relative.is_absolute() or ".." in relative.parts
            or len(relative.parts) < 2 or relative.parts[0] != "scripts"
            or relative.suffix != ".py"):
        return _result(success=False, error="only scripts/*.py inside a skill may run")
    try:
        skill = manager._skill_loader.find_skill(name)
        if not skill:
            return _result(success=False, error="skill not found")
        skill_dir = manager._skill_path(skill)
        if skill_dir is None:
            return _result(success=False, error="skill path escapes the skills directory")
        # 拒绝路径上任何符号链接，防止通过 scripts 或文件链接绕过入口限制。
        original = Path(skill["path"]).absolute()
        root = manager._skills_dir.absolute()
        for path in (original, *original.parents):
            if path == root:
                break
            if path.is_symlink():
                return _result(success=False, error="symlinked skill paths are not allowed")
        script = skill_dir
        for part in relative.parts:
            script = script / part
            if script.is_symlink():
                return _result(success=False, error="symlinked scripts are not allowed")
        script = script.resolve()
        if not script.is_relative_to(skill_dir / "scripts") or not script.is_file():
            return _result(success=False, error="script not found inside skill scripts directory")
        return _run(script, args, skill_dir, timeout)
    except (OSError, ValueError, RuntimeError) as exc:
        return _result(success=False, error=f"script execution failed: {type(exc).__name__}")


registry.register(
    name="skill_run",
    description=(
        "Run a trusted Python script under a discovered skill's scripts directory. "
        "Read the skill instructions first. Pass arguments as separate strings, not a shell command. "
        "Uses the current Python environment and skill directory as cwd. "
        "NOT a sandbox: scripts have current user permissions and inherited environment. "
        "Do not execute untrusted or newly generated code without user approval. "
        "Returns JSON with stdout, stderr, exit_code, timed_out and truncated; "
        "execution stops after timeout or 32768 output bytes. Never auto-install dependencies."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact skill name"},
            "file_path": {"type": "string", "description": "Skill-relative scripts/*.py path"},
            "args": {"type": "array", "items": {"type": "string"}, "maxItems": 100},
            "timeout": {"type": "integer", "minimum": 1, "maximum": 120, "default": 30},
        },
        "required": ["name", "file_path"],
        "additionalProperties": False,
    },
    handler=skill_run,
    category="execution",
)