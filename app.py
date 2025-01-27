from ia_agent import IAAgent
from chatbox_ui import ChatboxUI

# Initialize the agent with a default model
agent = IAAgent(max_history=20)
models = agent.list_available_models()
if models:
    agent.model_name = models[0]  # Set the first model as default

# Initialize the UI
chatbox_ui = ChatboxUI(agent)

# Build and launch the interface
interface = chatbox_ui.build_interface()
interface.launch()
