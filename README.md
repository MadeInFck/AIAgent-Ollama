# AI Agent Ollama

## Local AI Agent based on local models powered by [Ollama](https://ollama.com)


### Featuring:
- Agent runs in terminal in main branch, or GUI powered by Gradio on gui-agent branch
- Streaming generated response
- Chat history (20 last messages) to provide context to the LLM
- Menu for LLM selection among LLMs pulled locally in Ollama app
- At prompt, type "exit" or "quit" to stop the discussion and leave app / GUI just use UI
- During generation, type "stop" and press enter to stop generation and streaming and return to prompt (threaded process)
- Model can be changed at prompt without resetting chat history (type "!change_model")
