import dataclasses


@dataclasses.dataclass
class LlmResponseChunk:
    content: str