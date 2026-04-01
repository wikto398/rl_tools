from pathlib import Path
import enum
import re


class PyGodotResource:
    class VariableType(enum.Enum):
        ARRAY = "Array"
        DICTIONARY = "Dictionary"
        STRING = "String"
        INT = "int"
        FLOAT = "float"
        RESOURCE = "Resource"
        UNKNOWN = "Unknown"

    def __init__(self, path: str | Path, *, create_backup: bool = True):
        self.path = Path(path)
        self.variables = {}
        self.content = self._parse()
        self._get_variables()

    def _parse(self):
        with self.path.open() as f:
            content = f.read()
        return content

    def _get_variables(self):
        pattern = r"(\[resource\][\s\S]*?)(?=\n\[|$)"
        resource_section = re.search(pattern, self.content)
        if resource_section:
            resource_content = resource_section.group(1)
            value = None
            key = None
            is_parsing_multiline = False
            for line in resource_content.splitlines():
                if "=" in line:
                    if is_parsing_multiline and key is not None and value is not None:
                        self.set_variable(key.strip(), value.strip())
                        value = None
                        key = None
                        is_parsing_multiline = False
                    key, value = line.split("=", 1)
                    if self._get_type(value.strip()) in [
                        self.VariableType.ARRAY,
                        self.VariableType.DICTIONARY,
                    ]:
                        is_parsing_multiline = True
                        continue
                    self.set_variable(key.strip(), value.strip())
                elif is_parsing_multiline and value is not None:
                    value += line.strip()
                    print(f"Updated value for {value}")

    def _get_type(self, value: str):
        if value.startswith("[") and value.endswith("]"):
            return self.VariableType.ARRAY
        elif value.startswith("Dictionary"):
            return self.VariableType.DICTIONARY
        elif value.startswith("ExtResource") or value.startswith("SubResource"):
            return self.VariableType.RESOURCE
        elif value.startswith('"') and value.endswith('"'):
            return self.VariableType.STRING
        elif value.isdigit():
            return self.VariableType.INT
        elif re.match(r"^\d+\.\d+$", value):
            return self.VariableType.FLOAT
        else:
            return self.VariableType.UNKNOWN

    def _format_type(self, value: str):
        var_type = self._get_type(value)
        match var_type:
            case self.VariableType.ARRAY:
                return value
            case self.VariableType.DICTIONARY:
                return self._unpack_dictionary(value)
            case self.VariableType.STRING:
                return value.strip('"')
            case self.VariableType.INT:
                return int(value)
            case self.VariableType.FLOAT:
                return float(value)
            case self.VariableType.RESOURCE:
                return value
            case _:
                return value

    def set_variable(self, key: str, value):
        if key in self.variables:
            formatted_value = self._format_type(str(value))
            self.variables[key]["value"] = formatted_value
        else:
            self.variables[key] = {
                "type": self._get_type(str(value)),
                "value": self._format_type(str(value)),
                "content": str(value),
            }

    def _unpack_dictionary(self, value: str):
        dict_pattern = re.compile(r"\(\{(.+?)\}\)")
        match = dict_pattern.search(value)
        if match:
            dict_str = match.group(1)
            result = {}
            for item in dict_str.split(","):
                k, v = item.split(":")
                result[k.strip()] = self._format_type(v.strip())
            return result
        return {}

    def save(self):
        with self.path.open("w") as f:
            f.write(self.content)


if __name__ == "__main__":
    resource = PyGodotResource("resources/buildings/production/wood/Sawmill.tres")
    print(resource.content)
    print(resource.variables)
