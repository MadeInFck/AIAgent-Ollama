import ollama
import sys
import threading
import select

class IAAgent:
    def __init__(self, max_history=20):
        self.model_name = self.choose_model()  # Choisir le modèle initial
        self._preload_model()  # Précharger le modèle initial
        self.stop_event = threading.Event()  # Événement pour signaler l'interruption
        self.conversation_history = []  # Historique de la conversation
        self.max_history = max_history  # Nombre maximal de messages à conserver

    def _preload_model(self):
        """Précharge le modèle pour éviter le temps de chargement lors de la première utilisation."""
        try:
            ollama.generate(model=self.model_name, prompt="Préchargement du modèle.")
            print(f"Modèle {self.model_name} préchargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du préchargement du modèle: {e}")

    def _update_conversation_history(self, role, content):
        """Ajoute un message à l'historique et conserve uniquement les derniers messages."""
        self.conversation_history.append({"role": role, "content": content})
        # Ne conserver que les derniers messages
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def _generate_context(self):
        """Génère le contexte en concaténant les derniers messages."""
        return "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history]
        )

    def generate_response(self, prompt, stream=True):
        """Génère une réponse en tenant compte du contexte de la conversation."""
        try:
            # Ajouter le message de l'utilisateur à l'historique
            self._update_conversation_history("user", prompt)

            # Construire le contexte
            context = self._generate_context()

            if stream:
                # Streaming de la réponse
                print("IA: ", end="", flush=True)  # Commencer l'affichage sans saut de ligne
                response = ""
                for chunk in ollama.generate(
                    model=self.model_name,
                    prompt=f"{context}\nuser: {prompt}",  # Utiliser le contexte
                    stream=True
                ):
                    if self.stop_event.is_set():  # Vérifier si l'utilisateur a demandé l'interruption
                        print("\nGénération interrompue.", end="\n\n", flush=True)
                        self.stop_event.clear()  # Réinitialiser l'événement
                        return
                    word = chunk['response']
                    print(word, end="", flush=True)  # Afficher mot par mot
                    response += word
                # Ajouter la réponse de l'IA à l'historique
                self._update_conversation_history("assistant", response)
                print("\n", end="", flush=True)  # Saut de ligne après la fin du streaming
            else:
                # Réponse en une seule fois
                response = ollama.generate(
                    model=self.model_name,
                    prompt=f"{context}\nuser: {prompt}"  # Utiliser le contexte
                )['response']
                # Ajouter la réponse de l'IA à l'historique
                self._update_conversation_history("assistant", response)
                print(f"IA: {response}")
        except Exception as e:
            print(f"Erreur lors de la génération de la réponse: {e}")

    def list_available_models(self):
        """Liste les modèles disponibles localement."""
        try:
            models = ollama.list()
            return [model['model'] for model in models['models']]
        except Exception as e:
            print(f"Erreur lors de la récupération des modèles disponibles: {e}")
            return []

    def choose_model(self):
        """Permet à l'utilisateur de choisir un modèle parmi ceux disponibles."""
        models = self.list_available_models()
        if not models:
            print("Aucun modèle disponible localement.")
            sys.exit(1)

        print("Modèles disponibles localement:")
        for idx, model in enumerate(models):
            print(f"{idx + 1}. {model}")

        choice = int(input("Choisissez un modèle par son numéro: ")) - 1
        if 0 <= choice < len(models):
            return models[choice]
        else:
            print("Choix invalide. Utilisation du premier modèle par défaut.")
            return models[0]

    def change_model(self):
        """Change le modèle en cours d'utilisation."""
        print("\nChangement de modèle...")
        new_model = self.choose_model()
        if new_model == self.model_name:
            print(f"Le modèle {self.model_name} est déjà sélectionné.")
            return

        self.model_name = new_model
        self._preload_model()  # Précharger le nouveau modèle
        print(f"Modèle changé avec succès. Nouveau modèle : {self.model_name}")

def main():
    """Fonction principale pour gérer l'interaction avec l'utilisateur."""
    # Paramètre : nombre maximal de messages à conserver (ici 20, mais modifiable)
    agent = IAAgent(max_history=20)

    print("\nCommandes spéciales:")
    print("  - 'exit' ou 'quit' : Quitter la conversation.")
    print("  - 'stop' : Interrompre la génération en cours.")
    print("  - '!change_model' : Changer de modèle en cours de conversation.")
    print("Appuyez sur Entrée pour commencer.")

    while True:
        try:
            prompt = input("Vous: ")
            if prompt.lower() in ['exit', 'quit']:
                print("Fin de la conversation.")
                break
            elif prompt.lower() == '!change_model':
                agent.change_model()
                continue
            elif prompt.lower() == 'stop':
                if agent.stop_event.is_set():
                    print("Aucune génération en cours.")
                else:
                    agent.stop_event.set()
                continue

            # Réinitialiser l'événement d'interruption
            agent.stop_event.clear()

            # Lancer la génération de la réponse dans un thread séparé
            thread = threading.Thread(target=agent.generate_response, args=(prompt,), kwargs={"stream": True})
            thread.start()

            # Attendre que l'utilisateur tape 'stop' ou que la génération se termine
            while thread.is_alive():
                # Vérifier si une entrée clavier est disponible
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline().strip()
                    if user_input.lower() == 'stop':
                        agent.stop_event.set()  # Signaler l'interruption
                        break

            thread.join()  # Attendre que le thread se termine proprement

        except KeyboardInterrupt:
            print("\nConversation interrompue.")
            break

if __name__ == "__main__":
    main()
