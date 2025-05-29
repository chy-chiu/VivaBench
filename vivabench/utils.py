import re

import rapidjson
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def remove_json_markdown(json_str: str):
    """
    Process input that might contain JSON with markdown formatting:
    1. If the input is already valid JSON, return it unchanged
    2. Try to extract valid JSON from between markdown tags
    3. If extraction fails, apply basic cleaning and return the result

    Returns: The extracted or cleaned JSON string
    """
    if isinstance(json_str, dict):
        return json_str
    # First check if the input is already valid JSON
    try:
        rapidjson.loads(json_str)
        return json_str  # Return unchanged if already valid
    except rapidjson.JSONDecodeError:
        pass

    # Try to extract JSON from between markdown tags
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, json_str)

    if matches:
        for match in matches:
            try:
                # Verify this is valid JSON
                rapidjson.loads(match)
                return match  # Return extracted JSON if valid
            except rapidjson.JSONDecodeError:
                continue

    # Fall back to basic cleaning
    cleaned = (
        json_str.replace("```json\n", "")
        .replace("```", "")
        .replace("```json", "")
        .strip()
    )
    return cleaned


def remove_json_markdown_enhanced(json_str: str):
    """
    Process input that might contain JSON with markdown formatting:
    1. If the input is already valid JSON, return it unchanged
    2. Try to extract valid JSON from between markdown tags
    3. Try to find valid JSON after phrases like "corrected version:"
    4. Scan for valid JSON objects, prioritizing those at the end of the string
    5. If all extraction methods fail, apply basic cleaning and return the result

    Returns: The extracted or cleaned JSON string
    """
    if isinstance(json_str, dict):
        return json_str

    # First check if the input is already valid JSON
    try:
        rapidjson.loads(json_str)
        return json_str  # Return unchanged if already valid
    except rapidjson.JSONDecodeError:
        pass

    # Try to extract JSON from between markdown tags
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(pattern, json_str)

    if matches:
        for match in matches:
            try:
                rapidjson.loads(match)
                return match  # Return extracted JSON if valid
            except rapidjson.JSONDecodeError:
                continue

    # Try to find valid JSON after common correction phrases
    correction_phrases = [
        "is the corrected version:",
        "here is the corrected version:",
        "corrected version:",
        "is the correct version:",
        "here is the correct version:",
        "correct version:",
        "corrected JSON:",
        "correct JSON:",
        "here is the correct JSON:",
    ]

    # Add variations with newlines or different formatting
    variations = []
    for phrase in correction_phrases:
        variations.extend([phrase, "\n" + phrase, phrase.capitalize()])

    for delimiter in variations:
        if delimiter in json_str:
            parts = json_str.split(delimiter, 1)
            if len(parts) > 1:
                candidate = parts[1].strip()
                try:
                    rapidjson.loads(candidate)
                    return candidate
                except rapidjson.JSONDecodeError:
                    pass

    # Scan the string for all potential JSON objects
    start_positions = [i for i, char in enumerate(json_str) if char == "{"]
    end_positions = [i for i, char in enumerate(json_str) if char == "}"]

    # Sort positions to prioritize finding JSON at the end
    start_positions.sort(reverse=True)

    valid_jsons = []

    for start in start_positions:
        valid_end_positions = [end for end in end_positions if end > start]
        valid_end_positions.sort()  # Try shortest valid strings first

        for end in valid_end_positions:
            candidate = json_str[start : end + 1]
            try:
                rapidjson.loads(candidate)
                valid_jsons.append((candidate, start))
                break  # Found valid JSON from this start position
            except rapidjson.JSONDecodeError:
                continue

    # Return the JSON that appears last in the string
    if valid_jsons:
        valid_jsons.sort(key=lambda x: x[1], reverse=True)
        return valid_jsons[0][0]

    # Fall back to basic cleaning
    cleaned = (
        json_str.replace("```json\n", "")
        .replace("```", "")
        .replace("```json", "")
        .strip()
    )
    return cleaned


def smart_capitalize(s):
    if len(s) <= 1:
        return s.upper()
    else:
        return s[0].upper() + s[1:]


def prettify(s):
    if not isinstance(s, str):
        return ""
    return smart_capitalize(s.replace("_", " "))


def normalize_key(x):
    x = x.lower().replace(" ", "_")
    if x.startswith("no_"):
        x = x.replace("no_", "")
    return x


def init_openrouter_chat_model(
    model_name: str, temperature: float, api_key: str, **kwargs
):
    """
    Initializes a chat model from OpenAI or OpenRouter.

    Args:
        model_identifier: String in the format "provider:model_name"
                          e.g., "openai:gpt-4o-mini"
                          e.g., "openrouter:anthropic/claude-3-opus-20240229"
        temperature: The sampling temperature.
        api_key: The API key for the specified provider.
        **kwargs: Additional arguments for the Chat model constructor.

    Returns:
        An instance of ChatOpenAI configured for the specified provider.
    """

    return ChatOpenAI(
        model_name=model_name,  # e.g., "anthropic/claude-3-opus-20240229"
        temperature=temperature,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,  # This is your OPENROUTER_API_KEY
        **kwargs,
    )


def init_ollama_chat_model(
    model_name: str,
    host: str = "localhost",
    port: int = 11434,
    temperature: float = 0.7,
    **kwargs,
):
    """
    Initializes a chat model pointing at a local Ollama server.

    Args:
        model_name:    Ollama model identifier, e.g. "gpt-4o" or a local ggml Q4_K_M model.
        host:          Ollama server host (defaults to "localhost").
        port:          Ollama server port (defaults to 11434).
        temperature:   Sampling temperature.
        **kwargs:      Any extra kwargs passed through to LangChain’s Ollama constructor.

    Returns:
        An instance of langchain.chat_models.Ollama configured to hit your local Ollama endpoint.
    """

    base_url = f"http://{host}:{port}"
    return ChatOllama(
        model=model_name, base_url=base_url, temperature=temperature, **kwargs
    )

def transform_agent_trace(input_text):
    # Extract the action, query, and reasoning
    lines = input_text.strip().split('\n')
    action_line = lines[0].strip()
    query_line = lines[1].strip()
    reasoning_lines = lines[2:]
    
    # Extract the action type
    action_type = action_line.replace('Action: ', '').strip()
    
    # Extract the query
    if "diagnosis" in action_line.lower():
        query = query_line.replace('Query: ', '').strip()
        # print(query)
        ddx = eval(query)
        query = ", ".join([f"(condition: {d.get('condition', d.get('diagnosis'))}, confidence: {d['confidence']})" for d in ddx])
        
    else:
        query = query_line.replace('Query: ', '').strip()
    
    # Extract the reasoning
    reasoning = ' '.join([line.replace('Reasoning: ', '') for line in reasoning_lines]).strip()
    
    # Format the output
    output = f"Agent: {reasoning}\n[{action_type.lower()}] {query}"
    
    return output
