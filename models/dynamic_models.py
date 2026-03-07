from pydantic import create_model, BaseModel
from typing import Dict, Any, Type, List, Optional, Union
from datetime import date

class DynamicModelGenerator:
    @staticmethod
    def map_type(type_str: str) -> Type:
        type_mapping = {
            "string": str,
            "str": str,
            "integer": int,
            "int": int,
            "float": float,
            "number": float,
            "boolean": bool,
            "bool": bool,
            "date": date,
            "any": Any,
            "dict": Dict[str, Any],
            "list": List[Any]
        }
        return type_mapping.get(type_str.lower(), str)

    @classmethod
    def create_pydantic_model(cls, model_name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
        fields = {}
        for field_name, field_def in schema.items():
            if isinstance(field_def, dict):
                if field_def.get("type") == "array" and "items" in field_def:
                    item_type = field_def["items"]
                    if isinstance(item_type, dict):
                        # Recursive call for nested objects in array
                        nested_model = cls.create_pydantic_model(f"{field_name}Item", item_type)
                        fields[field_name] = (List[nested_model], ...)
                    else:
                        fields[field_name] = (List[cls.map_type(item_type)], ...)
                elif field_def.get("type") == "object" and "properties" in field_def:
                    # Recursive call for nested objects
                    nested_model = cls.create_pydantic_model(field_name.capitalize(), field_def["properties"])
                    fields[field_name] = (nested_model, ...)
                else:
                    # Fallback for simple dict-based type definition
                    fields[field_name] = (cls.map_type(field_def.get("type", "string")), ...)
            else:
                # Direct string type mapping
                fields[field_name] = (cls.map_type(str(field_def)), ...)
        
        return create_model(model_name, **fields)
