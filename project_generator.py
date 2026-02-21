"""
Project Generator — Saves generated project files to disk with proper structure.
"""
import os
from typing import Dict

from config import OUTPUT_DIR


class ProjectGenerator:
    """Writes generated project files to the filesystem."""

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir

    def save_project(self, project_name: str, files: Dict[str, str]) -> str:
        """
        Save all generated files to disk.

        Args:
            project_name: Name of the project (used as folder name).
            files: Dict mapping filepath → file content.

        Returns:
            Absolute path to the created project directory.
        """
        project_dir = os.path.join(self.output_dir, self._sanitize(project_name))
        os.makedirs(project_dir, exist_ok=True)

        for filepath, content in files.items():
            full_path = os.path.join(project_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)

            print(f"    ✓ {filepath}")

        print(f"\n[✓] Project saved to: {project_dir}")
        return project_dir

    @staticmethod
    def _sanitize(name: str) -> str:
        """Sanitize project name for use as directory name."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")
