import gradio as gr

class ChatboxUI:
    def __init__(self, agent):
        self.agent = agent
        self.selected_model = None  # Track the selected model

    def generate_response(self, message, chat_history):
        """Generate a response and update the chat history."""
        if not self.agent.model_name:
            chat_history.append({"role": "assistant", "content": "No model selected. Please select a model first."})
            yield chat_history, ""
            return

        chat_history.append({"role": "user", "content": message})
        yield chat_history, ""

        assistant_response = ""
        for chunk in self.agent.generate_response(message, stream=True):
            assistant_response = chunk["content"]
            if len(chat_history) > 0 and chat_history[-1]["role"] == "user":
                chat_history.append({"role": "assistant", "content": assistant_response})
            else:
                chat_history[-1] = {"role": "assistant", "content": assistant_response}
            yield chat_history, ""

    def clear_history(self):
        """Clear the chat history."""
        self.agent.conversation_history = []
        return []

    def update_ui(self):
        """Update the UI elements based on the selected model."""
        models = self.agent.list_available_models()
        selected_model = self.agent.model_name or (models[0] if models else None)
        self.selected_model = selected_model
        return (
            gr.Dropdown(choices=models, value=selected_model),
            gr.Button(interactive=bool(selected_model)),
        )

    def build_interface(self):
        """Build the Gradio interface."""
        with gr.Blocks() as interface:
            gr.Markdown("### Conversational Agent with Ollama")

            with gr.Column():
                chatbot = gr.Chatbot(
                    label="Conversation",
                    bubble_full_width=False,
                    avatar_images=(
                        "https://via.placeholder.com/40/0078D7/FFFFFF?text=U",
                        "https://via.placeholder.com/40/505050/FFFFFF?text=A",
                    ),
                    type="messages",
                )

                message = gr.Textbox(
                    label="Your message",
                    placeholder="Type your message here...",
                    lines=1,
                    max_lines=1,
                )

                with gr.Row():
                    send_button = gr.Button("Send", interactive=bool(self.agent.model_name))
                    clear_button = gr.Button("Clear")

            with gr.Accordion("Model Settings", open=True):
                model_dropdown = gr.Dropdown(
                    choices=self.agent.list_available_models(),
                    label="Select Model",
                    value=self.agent.model_name,
                    allow_custom_value=True,
                )
                change_model_button = gr.Button("Change Model", interactive=bool(self.agent.model_name))
                model_status = gr.Textbox(label="Model Status", interactive=False)

            # Actions
            send_button.click(
                self.generate_response,
                inputs=[message, chatbot],
                outputs=[chatbot, message],
                queue=True
            )
            clear_button.click(
                self.clear_history,
                inputs=[],
                outputs=chatbot,
                queue=False
            )
            change_model_button.click(
                self.agent.change_model,
                inputs=[model_dropdown],
                outputs=[model_status],
                queue=False
            ).then(
                self.update_ui,
                inputs=[],
                outputs=[model_dropdown, send_button],
                queue=False
            )
            model_dropdown.change(
                lambda model: None,  # Do not return anything
                inputs=[model_dropdown],
                outputs=[],
            )

            # Send with the "Enter" key
            message.submit(
                self.generate_response,
                inputs=[message, chatbot],
                outputs=[chatbot, message],
                queue=True
            )

        return interface
