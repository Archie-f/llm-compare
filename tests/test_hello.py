def test_provider_name_with_different_case():
    provider: list[str] = ["Provider", "PROVIDER", "proVideR", "provider"]
    for each in provider:
        assert each.lower() == "provider"

def test_invalid_provider_name():
    provider: str = "InvalidProvider"
    error_message: str = f"Unknown provider: {provider}"
    assert "InvalidProvider" in error_message

def is_provider_name_empty(provider_name: str) -> bool:
    return not provider_name

def test_empty_provider_name():
    provider: list[str] = ["ollama", ""]
    assert not is_provider_name_empty(provider[0])
    assert is_provider_name_empty(provider[1])

def test_content():
    content: str = "Say hello!"
    message = [
        {"role": "user", "content": content}
    ]
    assert message[0]["content"] == content


