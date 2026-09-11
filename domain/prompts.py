from domain.llm.llm_request import LlmRequest
from pydantic import BaseModel
from abc import ABC, abstractmethod


class Prompt(ABC):

    @staticmethod
    @abstractmethod
    def build(user_prompt: str) -> LlmRequest:
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


class SynthesizePrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> Response:
        return Response.model_validate_json(response)

    @staticmethod
    def build(user_prompt: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=user_prompt,
            system_prompt="""
            Recibirás información procedente de varias fuentes.

            Combina la información en un único documento coherente.

            - Elimina información duplicada.
            - Agrupa información relacionada.
            - Si varias fuentes aportan información complementaria,
              combínala.
            - No inventes información.
            - Si existen contradicciones importantes, indícalas.
            - Conserva la información relevante para planificar
              o comprender el contenido.

            Devuelve exclusivamente un JSON válido:

            {
                "content": "..."
            }
            """
        )


class TextClassificationPrompt(Prompt):
    response_model = ClassificationResponse

    @staticmethod
    def parse(response: str) -> ClassificationResponse:
        return ClassificationResponse.model_validate_json(response)

    @staticmethod
    def build(user_prompt: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=user_prompt,
            system_prompt="""
            Clasifica el siguiente texto. Los tipos posibles son:
                - news
                - forum
                - travel
            En el caso de que no encuentres ningun tipo como estos, deberás establecer el tipo 'generic'
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
    def build(user_prompt: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=user_prompt,
            system_prompt="""
            Devuelve únicamente el contenido en formato Markdown.
            No incluyas JSON ni explicaciones adicionales.
            """
        )


class NewsPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> str:
        return response

    @staticmethod
    def build(user_prompt: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=user_prompt,
            system_prompt="""
            Resume objetivamente la noticia. Elimina sensacionalismo, opiniones y lenguaje emocional. 
            Conserva fechas, hechos, lugares, protagonistas, consecuencias y demás información objetiva.
            Devuelve exclusivamente un JSON válido:

            {
                "content": "..."
            }
            """
        )


class ForumPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> str:
        return response

    @staticmethod
    def build(user_prompt: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=user_prompt,
            system_prompt="""
            Sirve de apoyo en la interacción social en la comunidad: 
             - analiza las directrices de la web, como las reglas de comportamiento de la comunidad
             - elimina el lenguaje ambiguo o implícito, traduciéndolo a un lenguaje directo, neutral y estrictamente literal.
             - resume de forma muy clara y directa lo que está permitido y lo que no. 
            
            Devuelve exclusivamente un JSON válido:

            {
                "content": "..."
            }
            """
        )


class PlanningPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> str:
        return response

    @staticmethod
    def build(user_prompt: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=user_prompt,
            system_prompt="""
            Organiza la información en un itinerario paso a paso, lo más detallado y cronológico posible, 
            permitiendo a la persona anticipar cada momento. 
            
            Devuelve exclusivamente un JSON válido:

            {
                "content": "..."
            }
            """
        )


class GenericPrompt(Prompt):
    response_model = Response

    @staticmethod
    def parse(response: str) -> str:
        return response

    @staticmethod
    def build(user_prompt: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=user_prompt,
            system_prompt="""
            Resume el contenido eliminando el lenguaje figurado o irónico,  
            
            Devuelve exclusivamente un JSON válido:

            {
                "content": "..."
            }
            """
        )

class ContextSummaryPrompt(Prompt):
    @staticmethod
    def parse(response: str) -> str:
        return Response.model_validate_json(response).content

    @staticmethod
    def build(user_prompt: str) -> LlmRequest:
        return LlmRequest(
            user_prompt=user_prompt,
            system_prompt="""
            Responde a la pregunta del usuario usando el contexto proporcionado en la propia query del usuario.
            Sé descriptivo, pero mantente atado al contexto proporcionado.
            Devuelve exclusivamente un JSON válido:
            {
                "content": "..."
            }
            Si la respuesta no está contenida en el contexto, el valor de la
            respuesta debe ser "El texto no proporciona una respuesta a esta pregunta"
            
            """
        )
