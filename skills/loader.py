"""技能加载器（第 10 章）：发现技能、解析元数据并构建渐进式披露索引。"""

from pathlib import Path

import yaml


class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> list[dict]:
        """递归加载技能，返回名称、描述、正文、所在目录和版本。"""
        result = []
        # 固定顺序，避免文件系统遍历顺序变化影响系统提示词前缀。
        for skill_md in sorted(self.skills_dir.rglob("SKILL.md")):
            text = skill_md.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            if meta and meta.get("name"):
                result.append({
                    "name": meta["name"],
                    "description": meta.get("description", ""),
                    "body": body,
                    "path": skill_md.parent,
                    "version": meta.get("version", "1.0.0"),
                })
        return result

    def build_skills_index(self) -> str:
        """生成只含名称和描述的一级索引，不将技能正文放入系统提示词。"""
        all_skills = self.load_all()
        if not all_skills:
            return ""
        lines = ["Available skills (use skill_view to load full content):"]
        for skill in all_skills:
            desc = skill["description"][:200]
            lines.append(f"- **{skill['name']}**: {desc}")
        return "\n".join(lines)

    def find_skill(self, name: str) -> dict | None:
        """按名称精确查找技能，每次读取磁盘上的最新内容。"""
        for skill in self.load_all():
            if skill["name"] == name:
                return skill
        return None

    def _parse_frontmatter(self, text: str) -> tuple[dict, str]:
        """分离 YAML 元数据与正文；无效元数据返回空字典和原文。"""
        if not text.startswith("---"):
            return {}, text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text
        try:
            meta = yaml.safe_load(parts[1])
            if not isinstance(meta, dict):
                return {}, text
            return meta, parts[2].strip()
        except yaml.YAMLError:
            return {}, text