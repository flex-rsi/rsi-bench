"""Install the frozen libero-pro-agent panels (profile libero-agent-20261017)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_libero"))
from agent_install import install

if __name__ == "__main__":
    raise SystemExit(install("libero-pro-agent"))
