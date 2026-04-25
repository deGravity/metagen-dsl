"""Public entry point for the auto-generated API docs.

Run as a module:

    python -m metagen_dsl.docs --format llm
    python -m metagen_dsl.docs --format markdown

Or import directly:

    from metagen_dsl.docs import render_llm, render_markdown
"""

from ._docgen import render_llm, render_markdown, main, parse_docstring  # noqa: F401


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
