import abc

from domain.tools import ToolDefinition


class Tool(abc.ABC):
    @abc.abstractmethod
    def execute(self, **kwargs) -> str:
        ...

    @abc.abstractmethod
    def get_definition(self) -> ToolDefinition:
        pass