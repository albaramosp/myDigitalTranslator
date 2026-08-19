from domain.llm.llm_request import LlmRequest
from pydantic import BaseModel
from abc import ABC, abstractmethod


class Prompt(ABC):

    @staticmethod
    @abstractmethod
    def build(text) -> LlmRequest:
        ...

    @staticmethod
    @abstractmethod
    def parse(response):
        ...


class PromptFactory:
    @staticmethod
    def create(document_type: str) -> Prompt:
        if document_type == "news":
            return NewsPrompt()
        
        elif document_type == "forum":
            return ForumPrompt()
        
        elif document_type == "travel":
            return PlanningPrompt()
        
        else:
            return GenericPrompt()


class ClassificationResponse(BaseModel):
    type: str
    confidence: float


class Response(BaseModel):
    content: str


class TextClassificationPrompt(Prompt):
    response_model = ClassificationResponse

    @staticmethod
    def parse(response: str) -> ClassificationResponse:
        return ClassificationResponse.model_validate_json(response)

    @staticmethod
    def build(text: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=text,
            system_prompt="""
            Clasifica el siguiente texto. Los tipos posibles son:
                - news
                - forum
                - travel
                - generic
            
            Devuelve exclusivamente un JSON con este formato:
            {
                "type": "...",
                "confidence": 0.93
            }
            
            """
        )


class FormattingPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> Response:
        return Response.model_validate_json(response)

    @staticmethod
    def build(text: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=text,
            system_prompt="""
            Devuelve el contenido en un formato json como este:
            {
                "content": "contenido en formato markdown de alta calidad."
            }
            """
        )


class NewsPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> str:
        return response

    @staticmethod
    def build(text: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=text,
            system_prompt="""
            Resume objetivamente la noticia. Elimina sensacionalismo, opiniones y lenguaje emocional. 
            Conserva fechas, hechos, lugares, protagonistas, consecuencias y demás información objetiva.
            Devuelve el contenido en un formato json
            """
        )


class ForumPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> str:
        return response

    @staticmethod
    def build(text: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=text,
            system_prompt="""
            Sirve de apoyo en la interacción social en la comunidad: 
             - analiza las directrices de la web, como las reglas de comportamiento de la comunidad
             - elimina el lenguaje ambiguo o implícito, traduciéndolo a un lenguaje directo, neutral y estrictamente literal.
             - resume de forma muy clara y directa lo que está permitido y lo que no. 
            
            Devuelve el contenido en un formato json
            """
        )


class PlanningPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> str:
        return response

    @staticmethod
    def build(text: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=text,
            system_prompt="""
            Organiza la información en un itinerario paso a paso, lo más detallado y cronológico posible, 
            permitiendo a la persona anticipar cada momento. 
            
            Devuelve el contenido en un formato json
            """
        )


class GenericPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> str:
        return response

    @staticmethod
    def build(text: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=text,
            system_prompt="""
            Resume el contenido eliminando el lenguaje figurado o irónico,  
            
            Devuelve el contenido en un formato json
            """
        )
