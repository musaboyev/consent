"""Report credential-looking source leaks by filename only, never by value."""
from pathlib import Path
import re
import stat

root=Path(__file__).resolve().parents[1]
excluded={".venv","node_modules","dist",".git",".local","recordings","__pycache__"}
pattern=re.compile(rb"LLM_[A-Za-z0-9_-]{20,}")
leaks=[]
for path in root.rglob("*"):
    if not path.is_file() or path.name==".env" or any(part in excluded for part in path.relative_to(root).parts):continue
    if path.suffix.lower() in {".onnx",".task",".mp4",".jpg",".png",".pyc"}:continue
    if path.stat().st_size>2_000_000:continue
    if pattern.search(path.read_bytes()):leaks.append(str(path.relative_to(root)))
if leaks:raise SystemExit("Credential-looking content found in: "+", ".join(leaks))
env=root/".env"
if env.exists() and stat.S_IMODE(env.stat().st_mode)!=0o600:raise SystemExit("Backend .env must have mode 600")
assert ".env" in (root/".gitignore").read_text().splitlines()
assert ".local/" in (root/".gitignore").read_text().splitlines()
print("No credential-looking source/report content found; backend secret permissions and ignore rules checked.")
