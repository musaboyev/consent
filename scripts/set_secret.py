"""Interactive local key entry. No echo, shell argument, history, or key logging."""
import getpass
import os
from pathlib import Path
import re

def main():
    root=Path(__file__).resolve().parents[1]
    path=root/".env"
    key=getpass.getpass("Meta key (hidden): ").strip()
    if not key or any(c in key for c in "\r\n\x00"):
        raise SystemExit("Invalid empty or multiline key")
    existing=path.read_text() if path.exists() else (root/".env.example").read_text()
    line="MODEL_API_KEY="+key
    if re.search(r"^MODEL_API_KEY=.*$",existing,re.MULTILINE):
        existing=re.sub(r"^MODEL_API_KEY=.*$",lambda _:line,existing,flags=re.MULTILINE)
    else:
        existing+="\n"+line+"\n"
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(descriptor,"w") as file:
        file.write(existing)
    path.chmod(0o600)
    print("Backend key stored in ignored .env; value not displayed.")

if __name__=="__main__":main()
