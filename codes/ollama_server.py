import subprocess
import time

def start_server():
    """
    Starts the Ollama server in the background.
    Returns the subprocess.Popen object so it can be terminated later if needed.
    """
    # Launch the server in the background
    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Wait a few seconds to give the server time to start
    time.sleep(5)

    # Optional: check if the server is running by checking the process
    if process.poll() is None:
        print("Ollama server started successfully in the background.")
    else:
        print("Failed to start Ollama server.")
    
    return process

def stop_server(process):
    """Stops the Ollama server if it is running."""
    if process and process.poll() is None:
        process.terminate()
        process.wait()
        print("Ollama server stopped.")
