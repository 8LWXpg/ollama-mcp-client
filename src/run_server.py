import subprocess
import signal
import sys

# Define the commands
fastapi_cmd = ["uvicorn", "src.clients.api:app", "--host", "0.0.0.0", "--port", "8000"]
flask_cmd = ["uv", "run", "src/website/app.py"]

# Start both processes
p1 = subprocess.Popen(fastapi_cmd)
p2 = subprocess.Popen(flask_cmd)


# Handle Ctrl+C
def handle_exit(sig, frame):
    print("\nShutting down servers...")
    p1.terminate()  # Gracefully terminate FastAPI
    p2.terminate()  # Gracefully terminate Flask
    p1.wait()
    p2.wait()
    sys.exit(0)


# Bind signal handler
signal.signal(signal.SIGINT, handle_exit)

try:
    p1.wait()
    p2.wait()
except KeyboardInterrupt:
    handle_exit(None, None)
