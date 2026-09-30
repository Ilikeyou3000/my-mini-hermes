#!/usr/bin/env python3
"""Mini-Hermes CLI（第 15 章）：配置、组件接线与终端交互。"""

import logging
import sys
from pathlib import Path

import yaml
from openai import OpenAI

from agent import Agent
from compression import ContextCompressor
from memory.persistent import PersistentMemory
from memory.recall import SessionRecall
from memory.session_db import SessionDB
from prompt_builder import PromptBuilder
from skills.loader import SkillLoader
from tool_registry import registry

# 导入时注册本地已经实现的工具。
import skills.manager
import tools.memory_tool


def main():
    logging.basicConfig(level=logging.WARNING)
    root = Path(__file__).resolve().parent
    config_path = root / "config.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    client = OpenAI(
        api_key=config["model"]["api_key"],
        base_url=config["model"]["base_url"],
    )
    model = config["model"]["model"]
    max_tokens = config["model"].get("max_tokens", 400)
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    skills_dir = data_dir / "skills"
    skills_dir.mkdir(exist_ok=True)

    session_db = SessionDB(data_dir / "state.db")
    session_id = None
    try:
        persistent = PersistentMemory(data_dir)
        skill_loader = SkillLoader(skills_dir)
        recall = SessionRecall(
            session_db, client, model,
            max_tokens=config.get("aux_model", {}).get("max_tokens", 300),
        )
        tools.memory_tool.set_memory(persistent, recall)
        skills.manager.set_skill_loader(skill_loader, skills_dir)

        builder = PromptBuilder()
        system_prompt = builder.get_system_prompt(
            memory_block=persistent.load(),
            skills_index=skill_loader.build_skills_index(),
        )
        session_id = session_db.create_session(source="cli")
        # 本地会话表没有 system_prompt 字段，使用系统消息保存启动快照。
        session_db.append_message(session_id, "system", system_prompt)
        recall.current_session_id = session_id

        agent = Agent(
            client=client,
            model=model,
            system_prompt=system_prompt,
            tools=registry.get_schemas(),
            tool_handlers=registry.get_handlers(),
            max_iterations=config.get("agent", {}).get("max_iterations", 15),
            max_tokens=max_tokens,
        )
        agent.session_db = session_db
        agent.session_id = session_id
        learning = config.get("learning", {})
        agent.configure_learning(
            memory_nudge=learning.get("memory_nudge_interval", 5),
            skill_nudge=learning.get("skill_nudge_interval", 8),
        )
        agent.set_compressor(ContextCompressor(
            client=client, model=model,
            max_context_tokens=32000, max_tokens=max_tokens,
        ))

        print("╔══════════════════════════════════════╗")
        print("║       Mini-Hermes Agent v0.1         ║")
        print("║  /mem /skills /model /sessions       ║")
        print("║  exit to quit                        ║")
        print("╚══════════════════════════════════════╝")
        print(f"  Model: {model}")
        print(f"  Session: {session_id[:8]}...")
        mem_status = "loaded" if persistent.load() else "empty"
        print(f"  Memory: {mem_status} | Skills: {len(skill_loader.load_all())}")
        print()

        try:
            _repl(agent, client, persistent, skill_loader, recall)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
    finally:
        try:
            if session_id is not None:
                session_db.end_session(session_id)
                print(f"Session {session_id[:8]} saved.")
        finally:
            session_db.close()


def _repl(agent, client, persistent, skill_loader, recall):
    while True:
        user_input = input("\033[1;32myou >\033[0m ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            print("Goodbye!")
            break

        if user_input == "/mem":
            print("\n\033[1;33m── Memory ──\033[0m")
            print(persistent.read_memory())
            print("\n\033[1;33m── User Profile ──\033[0m")
            print(persistent.read_user_profile())
            print()
            continue

        if user_input == "/skills":
            all_skills = skill_loader.load_all()
            if not all_skills:
                print("No skills yet.\n")
            else:
                print(f"\n\033[1;33m── {len(all_skills)} Skills ──\033[0m")
                for skill in all_skills:
                    print(f"  {skill['name']}: {skill['description'][:60]}")
                print()
            continue

        if user_input == "/model":
            print("\033[2m  fetching models...\033[0m", end="", flush=True)
            try:
                model_ids = [model.id for model in client.models.list().data]
            except Exception as exc:
                print(f"\r  Error listing models: {exc}\n")
                continue
            print("\r" + " " * 40 + "\r", end="")
            if not model_ids:
                print("No models available.\n")
                continue
            options = [f"{mid}  ← active" if mid == agent.model else mid
                       for mid in model_ids]
            default = model_ids.index(agent.model) if agent.model in model_ids else 0
            try:
                # 只有交互选择模型时才需要终端菜单依赖。
                from pick import pick

                _, idx = pick(
                    options, "Select model (↑↓ to move, Enter to select, Ctrl-C to cancel):",
                    default_index=default,
                )
                agent.set_model(model_ids[idx])
                print(f"  Switched to \033[1m{agent.model}\033[0m "
                      f"({type(agent._strategy).__name__})\n")
            except ImportError:
                print("  Install the menu dependency: python3 -m pip install pick\n")
            except (KeyboardInterrupt, EOFError):
                print("  Cancelled.\n")
            continue

        if user_input == "/sessions":
            query = input("Search query: ").strip()
            if query:
                print(recall.recall(query) or "No results.\n")
            continue

        print("\033[2m  thinking...\033[0m", end="", flush=True)
        response = agent.run(user_input)
        print("\r" + " " * 40 + "\r", end="")
        print(f"\033[1;36mhermes >\033[0m {response}\n")


if __name__ == "__main__":
    main()