"""
build_vault.py  —  Obsidian vault → vault.js converter
-------------------------------------------------------
Run from your repo root (where index.html lives):

    python build_vault.py

It walks ALL subdirectories looking for .md files, preserves
the folder hierarchy as the note's `folder` field, and writes
vault.js in the same directory.

Folder structure example:
  notes/
    ML/
      pca-svd.md
    OR/
      simplex.md
    poetry.md

The folder field will be "ML", "OR", and "notes" respectively.
Root .md files (next to index.html) are put in folder "Notes".
"""

import os
import re
import json

# ── Config ─────────────────────────────────────────────────
# Directories to scan (relative to this script's location).
# Add any folder names you use.
NOTE_DIRS = ["notes", "vault", "."]   # tries each in order; "." = root md files

# Files/folders to always skip
SKIP_FILES = {"vault.js", "README.md", "readme.md"}
SKIP_DIRS  = {".git", ".obsidian", "node_modules", "__pycache__", ".github"}
# ───────────────────────────────────────────────────────────


def slugify(title):
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def parse_yaml(yaml_str):
    metadata = {}
    current_key = None
    for line in yaml_str.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("-") and current_key is not None:
            val = line[1:].strip().strip("\"'")
            if current_key in metadata:
                if isinstance(metadata[current_key], list):
                    metadata[current_key].append(val)
                else:
                    metadata[current_key] = [metadata[current_key], val]
            else:
                metadata[current_key] = [val]
            continue
        if ":" in line:
            key, _, rest = line.partition(":")
            key = key.strip()
            val = rest.strip()
            if val.startswith("[") and val.endswith("]"):
                val_list = [v.strip().strip("\"'") for v in val[1:-1].split(",") if v.strip()]
                metadata[key] = val_list
            elif val == "":
                metadata[key] = []
            else:
                metadata[key] = val.strip("\"'")
            current_key = key
    return metadata


def collect_md_files(base_dir):
    """
    Walk base_dir recursively. Returns list of (filepath, folder_name).
    folder_name is the immediate parent dir relative to base_dir,
    or 'Notes' for files directly in base_dir.
    """
    results = []
    base_dir = os.path.abspath(base_dir)

    for root, dirs, files in os.walk(base_dir):
        # Prune skipped dirs in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for fname in sorted(files):
            if not fname.endswith(".md"):
                continue
            if fname in SKIP_FILES:
                continue
            filepath = os.path.join(root, fname)
            # Folder = relative path from base_dir to root
            rel = os.path.relpath(root, base_dir)
            if rel == ".":
                folder = "Notes"
            else:
                # Use the first path component as the top-level folder
                folder = rel.split(os.sep)[0]
                # Preserve ALL-CAPS acronyms (ML, OR, CS…); title-case mixed names
                if folder.isupper():
                    pass  # keep as-is: ML, OR, CS
                else:
                    folder = folder.replace("-", " ").replace("_", " ").title()
            results.append((filepath, folder))

    return results


def parse_note(filepath, folder):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    frontmatter = {}
    markdown_body = content

    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if fm_match:
        frontmatter = parse_yaml(fm_match.group(1))
        markdown_body = content[fm_match.end():]

    # Title priority: frontmatter → first H1 → filename
    title = frontmatter.get("title")
    if not title:
        h1 = re.search(r"^#\s+(.+)$", markdown_body, re.MULTILINE)
        title = h1.group(1).strip() if h1 else os.path.splitext(os.path.basename(filepath))[0]

    # Tags
    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]
    tags = [t.strip().lstrip("#") for t in tags if t.strip()]

    # Auto-tag from filename if no tags
    if not tags:
        name = os.path.basename(filepath).lower()
        if any(k in name for k in ("gradient", "optimizer", "adam", "momentum", "sgd")):
            tags = ["ml", "optimization"]
        elif any(k in name for k in ("pca", "svd", "eigenvalue")):
            tags = ["ml", "linear-algebra"]
        elif "poetry" in name or "urdu" in name:
            tags = ["poetry"]
        elif "photography" in name or "photo" in name:
            tags = ["photography"]
        elif any(k in name for k in ("cinema", "film", "movie")):
            tags = ["cinema"]
        elif any(k in name for k in ("or-", "simplex", "lp-", "qp-", "inventory")):
            tags = ["or", "optimization"]

    # Date
    date = str(frontmatter.get("date", ""))

    # Aliases
    aliases = frontmatter.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    aliases = [a.strip() for a in aliases if a.strip()]

    # Override folder from frontmatter if present
    if frontmatter.get("folder"):
        folder = frontmatter["folder"]

    return {
        "id": slugify(title),
        "title": title,
        "aliases": aliases,
        "folder": folder,
        "tags": tags,
        "date": date,
        "content": markdown_body.strip(),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    seen_ids = {}
    notes = []

    # Collect from configured note dirs
    collected = []
    for d in NOTE_DIRS:
        abs_d = os.path.join(script_dir, d)
        if os.path.isdir(abs_d):
            for fp, folder in collect_md_files(abs_d):
                # Avoid duplicates (e.g. "." includes subfolders already walked)
                if fp not in seen_ids:
                    seen_ids[fp] = True
                    collected.append((fp, folder))

    # Deduplicate by (folder, filename) — "." scan overlaps with named dirs
    seen_paths = set()
    for fp, folder in collected:
        norm = os.path.normpath(fp)
        if norm in seen_paths:
            continue
        seen_paths.add(norm)
        try:
            note = parse_note(fp, folder)
            notes.append(note)
            print(f"  ✓  [{note['folder']}]  {note['title']}")
        except Exception as e:
            print(f"  ✗  {fp}: {e}")

    # ── Resources: kept here; extend as needed ─────────────
    resources = [
        {
            "category": "Optimization",
            "title": "Distill: Why Momentum Really Works",
            "note": "A beautiful, interactive explanation of momentum — shows how it speeds up learning and how to analyse its dynamics.",
            "url": "https://distill.pub/2017/momentum/",
            "tags": ["optimization", "momentum", "interactive"],
        },
        {
            "category": "Optimization",
            "title": "An Overview of Gradient Descent Optimizers",
            "note": "Sebastian Ruder's classic post comparing all major gradient-descent variants. Required reading.",
            "url": "https://ruder.io/optimizing-gradient-descent/",
            "tags": ["gradient-descent", "optimizers", "ml"],
        },
        {
            "category": "Mathematics",
            "title": "Gilbert Strang — Linear Algebra (MIT OCW)",
            "note": "The definitive free course. Watch for intuition, not just computation. Essential for PCA/SVD.",
            "url": "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/",
            "tags": ["math", "linear-algebra", "mit"],
        },
        {
            "category": "Mathematics",
            "title": "3Blue1Brown — Essence of Linear Algebra",
            "note": "Geometric intuition for everything. Watch before any textbook.",
            "url": "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab",
            "tags": ["math", "youtube", "visual"],
        },
        {
            "category": "Machine Learning",
            "title": "Bishop — Pattern Recognition & ML",
            "note": "Dense but rewarding. The probabilistic graphical models chapter alone is worth it.",
            "url": "https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf",
            "tags": ["ml", "book", "probabilistic"],
        },
        {
            "category": "Computer Science",
            "title": "CMU 15-445 — Database Systems",
            "note": "Gold standard for database internals. Andy Pavlo's lectures are superb.",
            "url": "https://15445.courses.cs.cmu.edu/",
            "tags": ["dbms", "cmu", "cs"],
        },
        {
            "category": "Poetry",
            "title": "Rekhta — Urdu Poetry Library",
            "note": "The largest archive of Urdu poetry, with dictionaries and translations. Faiz, Mir, Ghalib — all here.",
            "url": "https://www.rekhta.org/",
            "tags": ["poetry", "urdu", "literature"],
        },
        {
            "category": "Cinema",
            "title": "Criterion Collection",
            "note": "Essays on films that deserve essays. The design sensibility alone is worth studying.",
            "url": "https://www.criterion.com",
            "tags": ["cinema", "essays", "aesthetics"],
        },
        {
            "category": "Tools",
            "title": "n8n — Workflow Automation",
            "note": "Self-hosted automation. Powerful for building bots and data pipelines without code.",
            "url": "https://n8n.io",
            "tags": ["automation", "tools", "no-code"],
        },
        {
            "category": "Photography",
            "title": "TFPS — IIT KGP Photography Club",
            "note": "The home base. Editorial photography, design, and the Summer Sprint leaderboard.",
            "url": "https://tfps.kgpian.in/",
            "tags": ["photography", "design", "iit"],
        },
    ]

    vault = {"notes": notes, "resources": resources}

    out_path = os.path.join(script_dir, "vault.js")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by build_vault.py — do not edit by hand\n")
        f.write("window.VAULT = ")
        json.dump(vault, f, indent=2, ensure_ascii=False)
        f.write(";\n")

    print(f"\n✅  vault.js written — {len(notes)} notes, {len(resources)} resources.")
    print(f"    Output: {out_path}")


if __name__ == "__main__":
    main()
