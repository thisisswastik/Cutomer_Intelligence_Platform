import os
import sys
import time
import socket
import webbrowser
import subprocess
from pathlib import Path

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def main():
    app_dir = Path(__file__).resolve().parent
    project_root = app_dir.parent
    db_path = project_root / "datasets" / "precomputed" / "olist_dashboard.db"
    precompute_script = app_dir / "backend" / "precompute.py"
    backend_script = app_dir / "backend" / "main.py"
    
    print("="*80)
    print("         Customer Intelligence Platform - Dashboard Launcher")
    print("="*80)
    print(f"Project path: {project_root}")
    
    # 1. Check if DB exists, if not run precompute
    if not db_path.exists():
        print("\n[INFO] Precomputed database not found. Starting initial data modeling...")
        print(f"Running: {sys.executable} {precompute_script}")
        try:
            subprocess.run([sys.executable, str(precompute_script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Initial precomputation failed with error code {e.returncode}.")
            sys.exit(1)
    else:
        print("\n[INFO] Found existing precomputed database.")

    # 2. Check if port 8000 is available
    port = 8000
    if is_port_in_use(port):
        print(f"\n[WARNING] Port {port} is already in use. The server might already be running.")
        print(f"Opening dashboard in your web browser: http://127.0.0.1:{port}")
        webbrowser.open(f"http://127.0.0.1:{port}")
        sys.exit(0)

    # 3. Start Uvicorn FastAPI server
    print(f"\n[INFO] Starting API server on http://127.0.0.1:{port}...")
    
    # Open browser in a separate process after 2 seconds
    def open_browser():
        time.sleep(2.0)
        print(f"\n[INFO] Opening browser to http://127.0.0.1:{port} ...")
        webbrowser.open(f"http://127.0.0.1:{port}")
        
    # Start thread or background subprocess to trigger browser open
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run Uvicorn command
    try:
        import uvicorn
        # Run main:app relative to backend directory
        sys.path.insert(0, str(app_dir / "backend"))
        uvicorn.run("main:app", host="127.0.0.1", port=port, log_level="info")
    except ImportError:
        print("\n[ERROR] uvicorn is not installed. Running via python subprocess...")
        try:
            subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)], cwd=str(app_dir / "backend"))
        except KeyboardInterrupt:
            print("\nServer stopped.")
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    main()
