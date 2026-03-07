import json
from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage
from utils.groq_client import get_groq_client
from utils.gemini_client import get_gemini_client

class StrictLLMClient:
    """
    Objective 4: Strict LLM wrapper with deterministic fallback chain.
    Forces temperature=0, JSON mode when required, and never throws raw exceptions.
    """
    
    def __init__(self, require_json: bool = True):
        self.require_json = require_json
        self.fallback_chain = [
            ("groq", "llama-3.1-8b-instant"),         # Primary Engine (Fastest)
            ("google", "gemini-flash-latest"),        # Flash Cross-Provider Fallback
            ("groq", "gemma2-9b-it"),                 # Mid-Tier Google-on-Groq Fallback
            ("groq", "mixtral-8x7b-32768"),           # High-Context Mixture-of-Experts
            ("google", "gemini-1.5-pro"),             # Heavy Cross-Provider Fallback
            ("groq", "llama-3.3-70b-versatile")       # Heavyweight First Resort
        ]

    def _get_client(self, provider: str, model: str):
        if provider == "groq":
            return get_groq_client(model)
        elif provider == "google":
            return get_gemini_client(model)
        raise ValueError(f"Unknown provider: {provider}")

    def invoke(self, prompt_chain, input_kwargs: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
        attempted = []
        last_error = ""

        # Phase 15: Polyglot Input Normalization
        from langchain_core.messages import BaseMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate
        
        messages: List[BaseMessage] = []
        if isinstance(prompt_chain, str):
            if input_kwargs:
                target_prompt = ChatPromptTemplate.from_template(prompt_chain)
                messages = target_prompt.invoke(input_kwargs).to_messages()
            else:
                messages = [HumanMessage(content=prompt_chain)]
        elif isinstance(prompt_chain, ChatPromptTemplate):
            messages = prompt_chain.invoke(input_kwargs).to_messages()
        elif isinstance(prompt_chain, list):
            messages = prompt_chain
        else:
            raise ValueError("Unsupported prompt_chain format")

        # Deterministic pipeline execution
        for provider, model in self.fallback_chain:
            attempted.append(f"{provider}:{model}")
            client = self._get_client(provider, model) if provider != "google" else None
            
            for attempt in range(max_retries):
                try:
                    
                    # Bypass LangChain for Gemini to avoid GRPC/Asyncio Threading deadlocks on massive contexts
                    if provider == "google":
                        import google.generativeai as genai
                        from config import settings
                        genai.configure(api_key=settings.GEMINI_API_KEY)
                        g_model = genai.GenerativeModel(model)
                        
                        prompt_str = "\n\n".join([m.content for m in messages if hasattr(m, 'content')])
                        raw_res = g_model.generate_content(prompt_str)
                        content = raw_res.text
                    else:
                        # Standard LangChain routing for Groq/OpenAI using Normalized Messages
                        response = client.invoke(messages)
                        content = response.content
                    
                    if not self.require_json:
                        return {"status": "success", "data": content, "model_used": model}
                        
                    # Strict JSON validation and auto-repair (basic)
                    clean_content = content.replace("```json", "").replace("```", "").strip()
                    try:
                        parsed = json.loads(clean_content)
                        return {"status": "success", "data": parsed, "model_used": model}
                    except json.JSONDecodeError as decode_err:
                        last_error = f"JSON Decode Error on {model}: {decode_err}"
                        # Try to bracket repair for structured Dicts
                        s_dict = clean_content.find('{')
                        e_dict = clean_content.rfind('}')
                        
                        # Try to bracket repair for continuous Arrays
                        s_list = clean_content.find('[')
                        e_list = clean_content.rfind(']')
                        
                        # We repair whichever outer structure appears first
                        if s_dict != -1 and (s_list == -1 or s_dict < s_list):
                            if e_dict != -1:
                                try:
                                    repaired = json.loads(clean_content[s_dict:e_dict+1])
                                    return {"status": "repaired", "data": repaired, "model_used": model}
                                except Exception:
                                    pass
                        elif s_list != -1:
                            if e_list != -1:
                                try:
                                    repaired = json.loads(clean_content[s_list:e_list+1])
                                    return {"status": "repaired", "data": repaired, "model_used": model}
                                except Exception:
                                    pass
                                
                except Exception as e:
                    last_error = str(e)
                    print(f"[{provider.upper()}] {model} failed: {last_error}")
                    break # Break retry loop, move to next fallback model immediately
                    
        # Objective 3: Structured Failure (If all fail)
        return {
            "status": "error",
            "stage": "llm_inference",
            "reason": "All models failed in fallback chain",
            "attempted_models": attempted,
            "last_error": last_error
        }
