import dataclasses


@dataclasses.dataclass
class LlmRequest:
    system_prompt: str
    user_prompt: str
    json_output: bool = True