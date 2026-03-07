from utils.gemini_client import get_gemini_client
from langchain_core.messages import HumanMessage
try:
    client = get_gemini_client("gemini-2.5-flash")
    client.transport = "rest"
    res = client.invoke([HumanMessage(content="Hello")])
    print(res.content)
except Exception as e:
    print(e)
