import os
import subprocess
import argparse
import time
import signal
import sys

# Process tracking
processes = []

def signal_handler(sig, frame):
    """Handle interrupt signals to gracefully shut down all processes."""
    print("\nShutting down all services...")
    for proc in processes:
        if proc.poll() is None:  # Check if process is still running
            proc.terminate()
    sys.exit(0)

def start_backend():
    """Start the LangServe backend API."""
    print("Starting LangServe backend API...")
    backend_process = subprocess.Popen(
        ["python", "-m", "backend.langserve_app"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes.append(backend_process)
    print(f"Backend API started with PID: {backend_process.pid}")
    return backend_process

def start_frontend():
    """Start the Streamlit frontend."""
    print("Starting Streamlit frontend...")
    frontend_process = subprocess.Popen(
        ["streamlit", "run", "frontend/streamlit_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes.append(frontend_process)
    print(f"Frontend started with PID: {frontend_process.pid}")
    return frontend_process

def check_health(process, name, timeout=60):
    """Monitor process health for the first few seconds."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if process.poll() is not None:
            # Process has terminated
            stdout, stderr = process.communicate()
            print(f"{name} process terminated unexpectedly!")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
        time.sleep(1)
    return True

def main():
    """Main entry point to start all services."""
    parser = argparse.ArgumentParser(description="Run LLM RAG SQL Application")
    parser.add_argument("--backend-only", action="store_true", help="Run only the backend API")
    parser.add_argument("--frontend-only", action="store_true", help="Run only the frontend UI")
    args = parser.parse_args()
    
    # Set up signal handling for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start backend if requested
        if not args.frontend_only:
            backend_proc = start_backend()
            # Give backend time to start before checking health
            time.sleep(5)
            if not check_health(backend_proc, "Backend", timeout=10):
                print("Backend failed to start properly. Exiting.")
                return
            print("Backend API is running.")
        
        # Start frontend if requested
        if not args.backend_only:
            # Give backend a moment to fully initialize
            if not args.frontend_only:
                time.sleep(2)
            frontend_proc = start_frontend()
            if not check_health(frontend_proc, "Frontend", timeout=10):
                print("Frontend failed to start properly. Exiting.")
                return
            print("Frontend is running.")
        
        # Keep main thread alive to receive keyboard interrupts
        while True:
            time.sleep(1)
            
            # Check if any process has terminated
            for i, proc in enumerate(processes):
                if proc.poll() is not None:
                    name = "Backend" if i == 0 else "Frontend"
                    stdout, stderr = proc.communicate()
                    print(f"{name} process terminated unexpectedly!")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    # Terminate all other processes
                    for p in processes:
                        if p != proc and p.poll() is None:
                            p.terminate()
                    return
                    
    except KeyboardInterrupt:
        print("\nShutting down...")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()

if __name__ == "__main__":
    main()