
from jsonschema import Draft202012Validator, ValidationError
def schema_valid(json_obj, schema) -> int:
    try:
        Draft202012Validator(schema).validate(json_obj)
        return 1
    except ValidationError:
        return 0
