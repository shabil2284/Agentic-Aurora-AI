from utils.strict_llm_client import StrictLLMClient
from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a json tool."),
    HumanMessage(content="test")
]
client = StrictLLMClient(require_json=False)
try:
    print("Invoking Client...")
    res = client.invoke(messages)
    print("Result:", res)
except Exception as e:
    print("Raw Exception:", e)
