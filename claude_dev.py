#!/usr/bin/env python3
"""
Claude Dev — terminal chat with access to your code.
Run from inside your project folder:  python claude_dev.py
"""

import os
import json
import anthropic

client = anthropic.Anthropic()
CWD = os.getcwd()

# ── Tools Claude can use ──────────────────────────────────────────────────────

def list_files(path="."):
    """Recursively list files, skipping common noise folders."""
    SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv", ".env",
            "dist", "build", ".next", ".mypy_cache", ".pytest_cache"}
    results = []
    base = os.path.join(CWD, path)
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            full = os.path.join(root, f)
            results.append(os.path.relpath(full, CWD))
    return "\n".join(results) if results else "No files found."

def read_file(path):
    full = os.path.join(CWD, path)
    if not os.path.exists(full):
        return f"Error: file '{path}' not found."
    try:
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.splitlines()
        numbered = "\n".join(f"{i+1:4}: {l}" for i, l in enumerate(lines))
        return f"--- {path} ({len(lines)} lines) ---\n{numbered}"
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path, content):
    full = os.path.join(CWD, path)
    os.makedirs(os.path.dirname(full), exist_ok=True) if os.path.dirname(full) else None
    try:
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✓ Written to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def run_tool(name, inputs):
    if name == "list_files":
        return list_files(inputs.get("path", "."))
    if name == "read_file":
        return read_file(inputs["path"])
    if name == "write_file":
        return write_file(inputs["path"], inputs["content"])
    return f"Unknown tool: {name}"

# ── Tool definitions for the API ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "list_files",
        "description": "List all files in the project directory (or a subdirectory). Use this to explore the project structure before reading specific files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to list. Defaults to '.' (project root)."}
            }
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file. Always read a file before modifying it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file."}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file. This overwrites the file completely. Use this to fix bugs, refactor code, or create new files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file."},
                "content": {"type": "string", "description": "Full content to write."}
            },
            "required": ["path", "content"]
        }
    }
]

SYSTEM = f"""You are an expert coding assistant with direct access to the user's project files.

Project directory: {CWD}

You have three tools:
- list_files: explore the project structure
- read_file: read any file
- write_file: write/overwrite a file

BEHAVIOR:
- When the user asks about a file or piece of code, read it first, then answer.
- When asked to fix or refactor code, read the file, make the changes, write it back, and explain what you changed.
- When unsure which file is relevant, list files first to orient yourself.
- Be concise. Don't repeat back the entire file unless asked.
- Always confirm after writing a file."""

# ── Agentic loop ──────────────────────────────────────────────────────────────

def chat(history):
    """Run one turn: may call tools multiple times until Claude gives a final answer."""
    messages = history[:]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages
        )

        # Collect text and tool uses from this response
        tool_uses = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # Print any text Claude produced before/alongside tool calls
        if text_parts:
            print(f"\n\033[92mClaude:\033[0m {' '.join(text_parts)}")

        # If no tool calls, we're done
        if not tool_uses or response.stop_reason == "end_turn":
            # Add Claude's final response to history
            messages.append({"role": "assistant", "content": response.content})
            return messages

        # Execute tool calls and feed results back
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool in tool_uses:
            print(f"\033[90m  [calling {tool.name}({json.dumps(tool.input)})]\033[0m")
            result = run_tool(tool.name, tool.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool.id,
                "content": result
            })

        messages.append({"role": "user", "content": tool_results})

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\033[1m\033[92m✦ Claude Dev\033[0m  \033[90m(project: {CWD})\033[0m")
    print("\033[90mClaude can read and edit your files. Type 'exit' to quit.\033[0m\n")

    history = []

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\033[90mBye!\033[0m")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("\033[90mBye!\033[0m")
            break

        history.append({"role": "user", "content": user_input})
        history = chat(history)
        print()

if __name__ == "__main__":
    main()
