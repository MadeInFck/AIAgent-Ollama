import ollama

class IAAgent:
    def __init__(self, max_history=20):
        self.model_name = None
        self.conversation_history = []
        self.max_history = max_history
        self._preload_model()

    def _preload_model(self):
        """Preload the model to avoid loading time during the first use."""
        if self.model_name:
            try:
                ollama.generate(model=self.model_name, prompt="Load model.")
                print(f"Model {self.model_name} successfully loaded.")
            except Exception as e:
                print(f"Error during model loading: {e}")

    def _update_conversation_history(self, role, content):
        """Add a message to the history and keep only the last messages."""
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def _generate_context(self):
        """Generate the context by concatenating the last messages."""
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history]
        )

    def generate_response(self, prompt, stream=True):
        if not self.model_name:
            yield {"role": "assistant", "content": "No model selected. Please select a model first."}
            return

        try:
            self._update_conversation_history("user", prompt)
            context = self._generate_context()

            if stream:
                response = ""
                for chunk in ollama.generate(
                    model=self.model_name,
                    prompt=f"{context}\nuser: {prompt}",
                    stream=True
                ):
                    response += chunk['response']
                    yield {"role": "assistant", "content": response}
                self._update_conversation_history("assistant", response)
            else:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=f"{context}\nuser: {prompt}"
                )['response']
                self._update_conversation_history("assistant", response)
                yield {"role": "assistant", "content": response}

        except Exception as e:
            print(f"Error during response generation: {e}")
            yield {"role": "assistant", "content": f"An error occurred: {e}"}

    def list_available_models(self):
        """List the models available locally."""
        try:
            models = ollama.list()
            return [model['model'] for model in models['models']]
        except Exception as e:
            print(f"Error while fetching local models: {e}")
            return []

    def change_model(self, new_model):
        """Change the currently used model."""
        if not new_model:
            return "No model selected."

        if new_model == self.model_name:
            return f"Model {self.model_name} is already selected."

        self.model_name = new_model
        self._preload_model()
        return f"Model changed to {self.model_name}."
