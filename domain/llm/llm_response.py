import dataclasses


@dataclasses.dataclass
class LlmResponse:
    content: str
    tokens: int