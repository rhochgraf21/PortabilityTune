import os
import re

class Function:
    def __init__(self, name):
        self.name = name

# Somewhere outside the loop, define this helper function:
def replace_at_position(content, replacement, position):
    return content[:position] + replacement + content[position + len(replacement):]

def get_function_location(content, function_name):
    function_def_regex = re.compile(r"\b" + re.escape(function_name) + r"\s*\([^)]*\)\s*\{")
    match = function_def_regex.search(content)
    if match:
        return match.start()
    return None

def process_folder(folder_path):
    source_string = ""

    # Build source_string by reading all .opencl files
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if ".opencl" in filename:
                with open(os.path.join(root, filename), 'r') as f:
                    content = f.read()
                source_string += content

    function_regex = re.compile(r"\b(\w+)\s*\([^)]*\)\s*\{")
    all_function_names = {match.group(1) for match in function_regex.finditer(source_string)}

    all_functions = [Function(name) for name in all_function_names]

    # Process each .opencl file for function replacement and macro insertion
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if ".opencl" in filename:
                with open(os.path.join(root, filename), 'r') as f:
                    content = f.read()

                for function in sorted(all_functions, key=lambda x: len(x.name), reverse=True):
                    print("function name processing: ", function.name)
                    if function.name not in ["if", "for", "while"]:
                        definition_location = get_function_location(content, function.name)
                        if definition_location:
                            content = replace_at_position(content, function.name.upper(), definition_location)

                        function_call_regex = re.compile(r"\b" + re.escape(function.name) + r"\s*\(")
                        for call_match in function_call_regex.finditer(content):
                            content = replace_at_position(content, function.name.upper(), call_match.start())

                        if definition_location or function_call_regex.search(content):
                            literal_opener = 'R"('
                            pos = content.find(literal_opener) + len(literal_opener)
                            macro_def = f"#define {function.name.upper()}2 {function.name}\n#define {function.name.upper()} CONCATENATE({function.name.upper()}2, PARAMS)\n"
                            content = content[:pos] + macro_def + content[pos:]

                with open(os.path.join(root, filename), 'w') as f:
                    f.write(content)

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    process_folder(folder_path)
