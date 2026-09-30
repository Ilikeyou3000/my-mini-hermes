"""技能管理（第 10-12 章）：导入即注册工具，启动时注入加载器。"""

import json
import shutil
from pathlib import Path

from tool_registry import registry


_skill_loader = None
_skills_dir = None


def set_skill_loader(loader, skills_dir):
    """注入加载器与技能根目录；传入 None 可清除依赖。"""
    global _skill_loader, _skills_dir
    _skill_loader = loader
    _skills_dir = Path(skills_dir) if skills_dir is not None else None


def _safe_path(base: Path, relative: str) -> Path | None:
    """拒绝绝对路径、上级跳转和解析后越界的符号链接。"""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        return None
    root = base.resolve()
    target = (root / path).resolve()
    if target == root or not target.is_relative_to(root):
        return None
    return root / path


def _skill_path(skill: dict) -> Path | None:
    path = Path(skill["path"])
    root = _skills_dir.resolve()
    target = path.resolve()
    if path.is_symlink() or target == root or not target.is_relative_to(root):
        return None
    return target


def skills_list(category: str = None) -> str:
    """仅列出元数据，category 沿用参考实现的路径子串过滤。"""
    if _skill_loader is None:
        return "Error: skills not initialized"
    all_skills = _skill_loader.load_all()
    if category:
        all_skills = [s for s in all_skills if category in str(s["path"])]
    result = [{
        "name": s["name"],
        "description": s["description"][:200],
        "version": s.get("version", "1.0.0"),
    } for s in all_skills]
    return json.dumps(result, indent=2) if result else "No skills found."


def skill_view(name: str, file_path: str = None) -> str:
    """按需读取完整技能或技能目录内的附属文件。"""
    if _skill_loader is None or _skills_dir is None:
        return "Error: skills not initialized"
    skill = _skill_loader.find_skill(name)
    if not skill:
        return f"Error: skill '{name}' not found"
    skill_dir = _skill_path(skill)
    target = _safe_path(skill_dir, file_path or "SKILL.md") if skill_dir else None
    if target is None:
        return "Error: path must stay within the skill directory"
    if not target.is_file():
        return f"Error: file '{file_path or 'SKILL.md'}' not found in skill '{name}'"
    return target.read_text(encoding="utf-8")


def skill_manage(action: str, name: str, content: str = None,
                 category: str = None, old_string: str = None,
                 new_string: str = None, replace_all: bool = False,
                 file_path: str = None, file_content: str = None) -> str:
    """创建、局部替换、重写、删除技能，以及写入和删除附属文件。"""
    if _skill_loader is None or _skills_dir is None:
        return "Error: skills not initialized"

    if action == "create":
        if not content:
            return "Error: content is required for create (full SKILL.md text)"
        if not content.startswith("---"):
            return "Error: SKILL.md must start with YAML frontmatter (---)"
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            return "Error: name must be a single directory name"
        parent = _safe_path(_skills_dir, category) if category else _skills_dir.resolve()
        skill_dir = _safe_path(parent, name) if parent else None
        target = _safe_path(skill_dir, "SKILL.md") if skill_dir else None
        if target is None:
            return "Error: path must stay within the skills directory"
        if target.exists():
            return f"Error: skill '{name}' already exists. Use 'patch' or 'edit'."
        skill_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        message = f"Skill '{name}' created at {skill_dir}"
    else:
        if action not in {"patch", "edit", "delete", "write_file", "remove_file"}:
            return f"Error: unknown action '{action}'"
        if action == "patch":
            if not old_string:
                return "Error: old_string is required for patch"
            if new_string is None:
                return "Error: new_string is required for patch"
        if action == "edit" and not content:
            return "Error: content is required for edit (full SKILL.md text)"
        if action == "write_file" and (not file_path or file_content is None):
            return "Error: file_path and file_content required for write_file"
        if action == "remove_file" and not file_path:
            return "Error: file_path required for remove_file"
        skill = _skill_loader.find_skill(name)
        if not skill:
            return f"Error: skill '{name}' not found"
        skill_dir = _skill_path(skill)
        if skill_dir is None:
            return "Error: path must stay within the skills directory"
        if action == "delete":
            shutil.rmtree(skill_dir)
            message = f"Skill '{name}' deleted"
        else:
            relative = "SKILL.md" if action == "edit" else (file_path or "SKILL.md")
            target = _safe_path(skill_dir, relative)
            if target is None:
                return "Error: path must stay within the skill directory"
            if action == "write_file":
                allowed = {"references", "templates", "scripts", "assets"}
                parts = Path(file_path).parts
                resolved_parts = target.resolve().relative_to(skill_dir).parts
                if (len(parts) < 2 or parts[0] not in allowed
                        or resolved_parts[0] not in allowed):
                    return f"Error: file must be under one of: {', '.join(sorted(allowed))}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(file_content, encoding="utf-8")
                message = f"Written {file_path} in skill '{name}'"
            elif action == "edit":
                target.write_text(content, encoding="utf-8")
                message = f"Skill '{name}' fully rewritten"
            elif not target.is_file():
                return f"Error: file not found: {file_path or target}"
            elif action == "patch":
                current = target.read_text(encoding="utf-8")
                count = current.count(old_string)
                if count == 0:
                    return f"Error: old_string not found in {target.name}"
                if count > 1 and not replace_all:
                    return (f"Error: old_string found {count} times. "
                            "Use replace_all=true or add more context.")
                updated = current.replace(old_string, new_string, -1 if replace_all else 1)
                target.write_text(updated, encoding="utf-8")
                message = f"Patched '{name}': {count} replacement(s)"
            else:
                target.unlink()
                message = f"Removed {file_path} from skill '{name}'"
    return json.dumps({"success": True, "message": message})


registry.register(
    name="skills_list",
    description="List all available skills with name and description (metadata only).",
    parameters={
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Optional category filter"},
        },
    },
    handler=skills_list,
    category="skills",
)

registry.register(
    name="skill_view",
    description="View the full content of a skill or a specific supporting file within it.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name"},
            "file_path": {
                "type": "string",
                "description": "Optional: path to a supporting file (e.g. references/api.md)",
            },
        },
        "required": ["name"],
    },
    handler=skill_view,
    category="skills",
)

registry.register(
    name="skill_manage",
    description=(
        "Manage skills (create, update, delete). Skills are procedural memory.\n"
        "Actions: create (full SKILL.md), patch (old_string/new_string), "
        "edit (full rewrite), delete, write_file, remove_file.\n"
        "After difficult tasks, offer to save as a skill. Confirm before creating/deleting."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "patch", "edit", "delete", "write_file", "remove_file"],
            },
            "name": {"type": "string", "description": "Skill name (lowercase, hyphens)"},
            "content": {"type": "string", "description": "Full SKILL.md for create/edit"},
            "category": {"type": "string", "description": "Category subdirectory for create"},
            "old_string": {"type": "string", "description": "Text to find (patch)"},
            "new_string": {"type": "string", "description": "Replacement text (patch)"},
            "replace_all": {"type": "boolean", "description": "Replace all occurrences (patch)"},
            "file_path": {"type": "string", "description": "Supporting file path"},
            "file_content": {"type": "string", "description": "Content for write_file"},
        },
        "required": ["action", "name"],
    },
    handler=skill_manage,
    category="skills",
)