from pathlib import Path
import subprocess
import sys

app = Path(__file__).parent / "src" / "streamlit_app.py"
raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))
