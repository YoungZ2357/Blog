# render_mermaid.py
# Replaces ```mermaid...``` fenced blocks in content/ with Hugo shortcode references,
# and renders each unique diagram to static/mermaid-svg/{hash}-light.svg and {hash}-dark.svg
# using mermaid-cli (mmdc).
#
# Run from the repo root in CI, BEFORE `hugo build`. Changes are made to the CI checkout
# only and are never committed back to the repository.
#
# Requirements: mmdc (npm install -g @mermaid-js/mermaid-cli), Python 3.8+

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MERMAID_RE = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)
SVG_DIR = Path("static/mermaid-svg")

# On Windows, npm installs CLI tools as .cmd wrappers; bare "mmdc" is not found
# by subprocess unless the shell is involved. Use mmdc.cmd directly on Windows.
MMDC = ["mmdc.cmd"] if sys.platform == "win32" else ["mmdc"]


def block_hash(body: str) -> str:
    return hashlib.md5(body.strip().encode()).hexdigest()[:10]


def render_svg(body: str, hash_str: str, theme: str) -> bool:
    out = SVG_DIR / f"{hash_str}-{theme}.svg"
    if out.exists():
        return True  # already rendered; same content hash = same output

    with tempfile.NamedTemporaryFile(
        suffix=".mmd", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(body)
        tmp = f.name

    try:
        result = subprocess.run(
            MMDC + [
                "-i", tmp,
                "-o", str(out),
                "--configFile", f"scripts/mermaid-config-{theme}.json",
                "--puppeteerConfigFile", "scripts/puppeteer-config.json",
                "--backgroundColor", "transparent",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"[WARN] mmdc failed ({hash_str}-{theme}): {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"[WARN] mmdc exception ({hash_str}-{theme}): {e}")
        return False
    finally:
        os.unlink(tmp)


def process_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    if "```mermaid" not in content:
        return

    def replace(m: re.Match) -> str:
        body = m.group(1)
        if not body.strip():
            print(f"[SKIP] empty mermaid block in {path}")
            return m.group(0)

        h = block_hash(body)
        ok_light = render_svg(body, h, "light")
        ok_dark = render_svg(body, h, "dark")

        if ok_light and ok_dark:
            return f'{{{{< mermaid-svg hash="{h}" >}}}}'

        print(f"[SKIP] render failed for a block in {path}, keeping original")
        return m.group(0)

    new_content = MERMAID_RE.sub(replace, content)
    if new_content != content:
        path.write_text(new_content, encoding="utf-8")
        print(f"[OK]   {path}")


def main() -> None:
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    for md_path in sorted(Path("content").rglob("*.md")):
        process_file(md_path)


if __name__ == "__main__":
    main()
