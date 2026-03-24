import time
import ollama

class OllamaAdapter:
    """Adapter for Ollama models to ensure consistent interface.

    Parameters
    - model_name: name of the Ollama model
    - temperature: decoding temperature (0.0 for deterministic)
    - options: optional dict forwarded to ollama.chat options
    """

    def __init__(self, model_name, temperature: float = 0.0, options: dict | None = None):
        self.model_name = model_name
        self.temperature = temperature
        self.options = options or {}

    def generate(self, prompt: str) -> tuple[str, float]:
        """
        Generate text given a prompt.
        Returns (response_text, latency_seconds)
        """
        start = time.time()
        try:
            opts = {"temperature": self.temperature, **(self.options or {})}
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=opts,
            )
            text = response["message"]["content"].strip()
        except Exception as e:
            print(f"Error with model {self.model_name}: {e}")
            return "", 0.0
        end = time.time()
        return text, round(end - start, 2)

    def get_decoding_params(self) -> dict:
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "options": dict(self.options or {}),
        }
