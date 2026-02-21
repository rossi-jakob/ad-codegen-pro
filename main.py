"""
Offline AI Code Generator — Main entry point.

Usage:
    python main.py                 # Interactive chat mode
    python main.py --download      # Download model for offline use (run once with internet)
"""
import argparse
import sys
import os

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory

from config import OUTPUT_DIR
from model_loader import ModelLoader
from rag_engine import RAGEngine
from code_generator import CodeGenerator
from project_generator import ProjectGenerator

console = Console()


def print_banner():
    console.print(
        Panel.fit(
            "[bold cyan]🤖 Offline AI Code Generator[/bold cyan]\n"
            "[dim]Powered by CodeLlama-7B · Fully Offline · RAG-Enhanced[/dim]\n\n"
            "Commands:\n"
            "  [green]/project <desc>[/green]  — Generate a full project\n"
            "  [green]/add <filepath>[/green]  — Add a file to knowledge base\n"
            "  [green]/clear[/green]           — Clear conversation history\n"
            "  [green]/quit[/green]            — Exit",
            border_style="cyan",
        )
    )


def handle_project_command(
    description: str,
    generator: CodeGenerator,
    project_gen: ProjectGenerator,
):
    """Handle /project command to generate a full project."""
    console.print(f"\n[bold yellow]Generating project:[/bold yellow] {description}\n")

    files = generator.generate_project(description)

    # Display generated files
    for filepath, content in files.items():
        console.print(f"\n[bold green]── {filepath} ──[/bold green]")
        ext = os.path.splitext(filepath)[1].lstrip(".")
        lexer = ext if ext in ("py", "js", "ts", "html", "css", "json", "yaml", "md") else "text"
        console.print(Syntax(content, lexer, theme="monokai", line_numbers=True))

    # Save to disk
    project_name = description.split()[:3]
    project_name = "_".join(project_name).lower()
    project_gen.save_project(project_name, files)


def handle_add_command(filepath: str, rag: RAGEngine):
    """Handle /add command to index a file into the knowledge base."""
    if not os.path.isfile(filepath):
        console.print(f"[red]File not found: {filepath}[/red]")
        return

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    rag.add_code_file(filepath, content)
    console.print(f"[green]✓ Added {filepath} to knowledge base[/green]")


def main():
    parser = argparse.ArgumentParser(description="Offline AI Code Generator")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download model for offline use (requires internet once)",
    )
    args = parser.parse_args()

    # Initialize components
    loader = ModelLoader()

    if False: #args.download:
        console.print("[bold]Downloading model for offline use …[/bold]")
        loader.load()
        loader.save_locally()
        console.print("[bold green]✓ Model downloaded. You can now run fully offline.[/bold green]")
        return

    console.print("[dim]Loading model …[/dim]")
    loader.load()

    rag = RAGEngine()
    rag.initialize()

    generator = CodeGenerator(loader, rag)
    project_gen = ProjectGenerator()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_banner()

    # Set up input history
    history_file = os.path.join(os.path.dirname(__file__), ".chat_history")
    history = FileHistory(history_file)

    while True:
        try:
            user_input = prompt("\n💬 You: ", history=history).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # ── Handle commands ─────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/clear":
            generator.conversation_history.clear()
            console.print("[green]✓ Conversation cleared[/green]")
            continue

        if user_input.lower().startswith("/project "):
            description = user_input[9:].strip()
            if description:
                handle_project_command(description, generator, project_gen)
            else:
                console.print("[yellow]Usage: /project <project description>[/yellow]")
            continue

        if user_input.lower().startswith("/add "):
            filepath = user_input[5:].strip()
            handle_add_command(filepath, rag)
            continue

        # ── Normal code generation ──────────────────────────────
        console.print("[dim]Thinking …[/dim]")
        response = generator.generate(user_input)

        # Pretty-print code blocks
        if "```" in response:
            console.print(Markdown(response))
        else:
            console.print(f"\n[bold cyan]🤖 Assistant:[/bold cyan]\n{response}")


if __name__ == "__main__":
    main()
