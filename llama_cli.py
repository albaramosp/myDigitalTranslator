from domain.prompts.autism_adapter_prompt import AutismAdapterPrompt
from infrastructure.llm.local_client import load_chain, LlamaClient

question = input("Tell me, my friend > ")
request = AutismAdapterPrompt.build(question)
client = LlamaClient()
response = client.generate(request)

print(response.content)
