import ollama
import sys
import threading
import select

class IAAgent:
    def __init__(self, max_history=20):
        self.model_name = self.choose_model()  # Choose the initial model
        self._preload_model()  # Preload the initial model
        self.stop_event = threading.Event()  # Event to signal interruption
        self.conversation_history = []  # Conversation history
        self.max_history = max_history  # Maximum number of messages to keep

    def _preload_model(self):
        """Preload the model to avoid loading time during the first use."""
        try:
            ollama.generate(model=self.model_name, prompt="Load model.")
            print(f"Model {self.model_name} successfully loaded.")
        except Exception as e:
            print(f"Error during model loading: {e}")

    def _update_conversation_history(self, role, content):
        """Add a message to the history and keep only the last messages."""
        self.conversation_history.append({"role": role, "content": content})
        # Keep only the last messages
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def _generate_context(self):
        """Generate the context by concatenating the last messages."""
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history]
        )

    def generate_response(self, prompt, stream=True):
        """Generate a response considering the conversation context."""
        try:
            # Add the user's message to the history
            self._update_conversation_history("user", prompt)

            # Build the context
            context = self._generate_context()

            if stream:
                # Streaming the response
                print("IA: ", end="", flush=True)  # Start display without newline
                response = ""
                for chunk in ollama.generate(
                    model=self.model_name,
                    prompt=f"{context}\nuser: {prompt}",  # Use the context
                    stream=True
                ):
                    if self.stop_event.is_set():  # Check if the user requested interruption
                        print("\nGeneration stopped.", end="\n\n", flush=True)
                        self.stop_event.clear()  # Reset the event
                        return
                    word = chunk['response']
                    print(word, end="", flush=True)  # Display word by word
                    response += word
                # Add the AI's response to the history
                self._update_conversation_history("assistant", response)
                print("\n", end="", flush=True)  # Newline after streaming ends
            else:
                # Single response
                response = ollama.generate(
                    model=self.model_name,
                    prompt=f"{context}\nuser: {prompt}"  # Use the context
                )['response']
                # Add the AI's response to the history
                self._update_conversation_history("assistant", response)
                print(f"IA: {response}")
        except Exception as e:
            print(f"Error during response generation: {e}")

    def list_available_models(self):
        """List the models available locally."""
        try:
            models = ollama.list()
            return [model['model'] for model in models['models']]
        except Exception as e:
            print(f"Error while fetching local models: {e}")
            return []

    def choose_model(self):
        """Allow the user to choose a model from those available."""
        models = self.list_available_models()
        if not models:
            print("No model locally available.")
            sys.exit(1)

        print("Models locally available:")
        for idx, model in enumerate(models):
            print(f"{idx + 1}. {model}")

        choice = int(input("Select a model: ")) - 1
        if 0 <= choice < len(models):
            return models[choice]
        else:
            print("Invalid selection. First model in list will be used by default.")
            return models[0]

    def change_model(self):
        """Change the currently used model."""
        print("\nModel change...")
        new_model = self.choose_model()
        if new_model == self.model_name:
            print(f"Model {self.model_name} has already been selected.")
            return

        self.model_name = new_model
        self._preload_model()  # Preload the new model
        print(f"Model has been changed successfully. New model : {self.model_name}")

def main():
    """Main function to manage user interaction."""
    # Parameter: maximum number of messages to keep (here 20, but modifiable)
    agent = IAAgent(max_history=20)

    print("\nCommands you can use:")
    print("  - 'exit' or 'quit' : Quit conversation and app.")
    print("  - 'stop' : Stop generation in progress.")
    print("  - '!change_model' : Change model during discussion.")
    print("Press Enter to start.")

    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in ['exit', 'quit']:
                print("End of conversation.")
                break
            elif prompt.lower() == '!change_model':
                agent.change_model()
                continue
            elif prompt.lower() == 'stop':
                if agent.stop_event.is_set():
                    print("No ongoing generation.")
                else:
                    agent.stop_event.set()
                continue

            # Reset the interruption event
            agent.stop_event.clear()

            # Start response generation in a separate thread
            thread = threading.Thread(target=agent.generate_response, args=(prompt,), kwargs={"stream": True})
            thread.start()

            # Wait for the user to type 'stop' or for the generation to finish
            while thread.is_alive():
                # Check if keyboard input is available
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline().strip()
                    if user_input.lower() == 'stop':
                        agent.stop_event.set()  # Signal interruption
                        break

            thread.join()  # Wait for the thread to finish cleanly

        except KeyboardInterrupt:
            print("\nConversation interrupted.")
            break

if __name__ == "__main__":
    main()