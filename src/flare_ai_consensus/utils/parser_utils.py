import re


def parse_chat_response(response: dict) -> str:
    """Parse response from chat completion endpoint"""
    try:
        return response.get("choices", [])[0].get("message", {}).get("content", "")
    except IndexError as e:
        raise ValueError("No response from chat completion", e, response)


def extract_author(model_id: str) -> tuple[str, str]:
    """
    Extract the author and slug from a model_id.

    :param model_id: The model ID string.
    :return: A tuple (author, slug).
    """
    author, slug = model_id.split("/", 1)
    return author, slug

def extract_values(text) -> dict[str, str]:
    pattern = r'\{\s*"operation"\s*:\s*"([^"]+)",\s*"token_a"\s*:\s*"([^"]+)",\s*"token_b"\s*:\s*"([^"]+)",\s*"amount"\s*:\s*"([^"]+)"(?:,\s*"reason"\s*:\s*"([^"]*)")?\s*\}'

    match = re.search(pattern, text, re.DOTALL)

    if match:
        return {
            "operation": match.group(1),
            "token_a": match.group(2),
            "token_b": match.group(3),
            "amount": float(match.group(4)),
            "reason": match.group(5) if match.group(5) else "",
        }
    return {
        "operation": "",
        "token_a": "",
        "token_b": "",
        "amount": float(0),
        "reason": text,
    }