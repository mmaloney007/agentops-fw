from typing import Tuple, Dict, Any
from jsonschema import Draft7Validator, ValidationError

def validate_json(instance: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
    try:
        Draft7Validator(schema).validate(instance)
        return True, ""
    except ValidationError as e:
        return False, str(e)
