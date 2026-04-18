
def add_tool_directive(prompt: str):
    """
    Instruction to the llm to ouput tools in a specific format.

    Example:
    ```json
    {
        "function": {
          "name": "get_weather",
          "arguments": {
            "city": "Amsterdam",
            "units": "celsius"
          }
        }
    }
    ```
    """
    prompt_delta = """
    If you need to use a tool, output only a single json like so
     
    Given: Available Tools [get_weather]
    ''```json
    {
        "function": {
          "name": "get_weather",
          "arguments": {
            "city": "Amsterdam",
            "units": "celsius"
          }
        }
    }
    ```''
    
    Only output a single json in the exact format:
    
    ''```json
    {
        "function": {
            "name": "<tool_name>",
            "arguments": 
                {
                    <arguments also in json format that given in ur prompt>
                }
    }
    ```''
    Note: do not modify or expand the (file)paths you are given. 
    """

    return prompt + "\n" + prompt_delta