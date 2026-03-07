from utils.strict_llm_client import StrictLLMClient
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a JSON generator. Return {'test': 'success'}"),
    HumanMessage(content="Generate the JSON now.")
]

client = StrictLLMClient(require_json=True)
try:
    print("Invoking Client...")
    res = client.invoke(messages)
    print("Result:", res)
except Exception as e:
    print("Raw Exception:", e)
