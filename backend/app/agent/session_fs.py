"""Virtual filesystem for agent working memory — borrowed from DeepAgents.

DeepAgents gives every agent a virtual filesystem (ls/read/write/edit/glob/grep).
The agent uses it as working memory across turns: writing to /agent_memory.md,
recording decisions, tracking patterns.

Here we provide the same pattern: an in-memory filesystem per tutoring session.
The tutor can write observation notes, track grammar patterns, maintain a
"whiteboard" of current focus areas—all within the session lifecycle.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FileNode:
    """A file in the virtual filesystem."""
    path: str
    content: str = ""


class SessionFilesystem:
    """
    Per-session in-memory filesystem.

    Usage mirrors DeepAgents:
        fs.write("/learning_notes.md", "User confuses wa/ga")
        fs.read("/learning_notes.md")
        fs.ls("/")
        fs.append("/learning_notes.md", "Also misuse ni/de")
    """

    def __init__(self):
        self._files: Dict[str, FileNode] = {}
        # Pre-create key files with semantic names
        self.write("/learning_notes.md", "# Learning Notes\n")
        self.write("/error_patterns.md", "# Error Patterns\n")
        self.write("/focus_areas.md", "# Focus Areas\n")
        self.write("/session_state.md", "# Session State\n")

    # ── CRUD ───────────────────────────────────────────

    def read(self, path: str) -> Optional[str]:
        """Read file content, or None if not found."""
        node = self._files.get(self._normalize(path))
        return node.content if node else None

    def write(self, path: str, content: str) -> None:
        """Write (overwrite) a file."""
        norm = self._normalize(path)
        if norm in self._files:
            self._files[norm].content = content
        else:
            self._files[norm] = FileNode(path=norm, content=content)

    def append(self, path: str, line: str) -> None:
        """Append a line to a file."""
        existing = self.read(path) or ""
        self.write(path, existing.rstrip() + "\n" + line + "\n")

    def ls(self, directory: str = "/") -> List[str]:
        """List files under a directory. Simple prefix match."""
        norm_dir = self._normalize(directory).rstrip("/") + "/"
        return sorted([
            p for p in self._files
            if p.startswith(norm_dir) and p != norm_dir
        ])

    def glob(self, pattern: str) -> List[str]:
        """Glob-like simple matching (* wildcard)."""
        import fnmatch
        norm = self._normalize(pattern)
        return sorted([
            p for p in self._files
            if fnmatch.fnmatch(p, norm)
        ])

    def grep(self, pattern: str) -> Dict[str, List[str]]:
        """Search all files for lines containing pattern."""
        results: Dict[str, List[str]] = {}
        for path, node in self._files.items():
            matching_lines = [
                line for line in node.content.split("\n")
                if pattern.lower() in line.lower()
            ]
            if matching_lines:
                results[path] = matching_lines
        return results

    def exists(self, path: str) -> bool:
        return self._normalize(path) in self._files

    # ── Serialization ──────────────────────────────────

    def to_context_block(self) -> str:
        """Render the filesystem as a text block for LLM context.

        Mirrors DeepAgents: the content of files gets composed into the
        system prompt so the agent sees its own notes.
        """
        blocks = []
        for path, node in sorted(self._files.items()):
            content = node.content.strip()
            if content and not content.startswith("# "):  # skip empty template files
                blocks.append(f"<file path=\"{path}\">\n{content}\n</file>")
            elif content:
                # Include non-empty template files too
                lines = content.strip().split("\n")
                if len(lines) > 1:  # More than just the title
                    blocks.append(f"<file path=\"{path}\">\n{content}\n</file>")
        return "\n".join(blocks) if blocks else ""

    # ── Helpers ────────────────────────────────────────

    @staticmethod
    def _normalize(path: str) -> str:
        """Ensure paths start with / and have no trailing slash."""
        p = path.strip()
        if not p.startswith("/"):
            p = "/" + p
        return p.rstrip("/")

    def __repr__(self) -> str:
        return f"SessionFilesystem({len(self._files)} files)"
