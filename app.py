import subprocess
import time
import requests
from chatbox_ui import ChatboxUI
from ia_agent import IAAgent

def check_and_start_ollama():
    """Check if Ollama is running, and start it if necessary."""
    try:
        # Try to connect to the Ollama server
        response = requests.get("http://localhost:11434", timeout=5)
        if response == "Ollama is running%  ":
            print("Ollama is already running")
            return True
    except requests.exceptions.RequestException:
        print("Ollama is not running. Attempting to start it...")
        try:
            # Start Ollama in a subprocess
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait for the server to start
            time.sleep(5)  # Adjust the delay as needed
            print("Ollama server started successfully.")
            return True
        except Exception as e:
            print(f"Failed to start Ollama: {e}")
            return False
    return False

def main():
    # Check and start Ollama before initializing the app
    if not check_and_start_ollama():
        print("Unable to start Ollama. Exiting...")
        return

    # Initialize the agent and the UI
    agent = IAAgent()  # Replace with your actual agent class
    chatbox_ui = ChatboxUI(agent)

    # Build and launch the Gradio interface
    interface = chatbox_ui.build_interface()
    interface.launch()

if __name__ == "__main__":
    main()
