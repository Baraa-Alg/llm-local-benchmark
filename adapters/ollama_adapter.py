import re
import time
import ollama

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


class OllamaAdapter:
    """Adapter for Ollama models to ensure consistent interface.

    Parameters
    - model_name: name of the Ollama model
    - temperature: decoding temperature (0.0 for deterministic)
    - options: optional dict forwarded to ollama.chat options
    - system_prompt: optional system message prepended to every request
    """

    def __init__(
        self,
        model_name,
        temperature: float = 0.0,
        options: dict | None = None,
        system_prompt: str | None = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.options = options or {}
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> tuple[str, float]:
        """
        Generate text given a prompt.
        Returns (response_text, latency_seconds)
        """
        start = time.time()
        try:
            opts = {"temperature": self.temperature, **(self.options or {})}
            # `think` must be a top-level kwarg to ollama.chat(), not inside options
            think = opts.pop("think", None)
            no_think = opts.pop("no_think", None)

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            user_content = f"/no_think\n{prompt}" if no_think else prompt
            messages.append({"role": "user", "content": user_content})

            chat_kwargs = dict(
                model=self.model_name,
                messages=messages,
                options=opts,
            )
            if think is not None:
                chat_kwargs["think"] = think
            response = ollama.chat(**chat_kwargs)
            raw = response["message"]["content"]
            text = _THINK_RE.sub("", raw).strip()
            if not text:
                # Model put entire answer inside <think> block — extract last non-empty line
                m = _THINK_RE.search(raw)
                if m:
                    lines = [l.strip() for l in m.group(1).splitlines() if l.strip()]
                    text = lines[-1] if lines else raw.strip()
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
            "system_prompt": self.system_prompt,
        }
