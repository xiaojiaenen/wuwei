"""Normalize OpenAI usage without counting detail tokens a second time."""


def parse_openai_usage(usage, *, responses=False):
    def field(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    input_name, output_name = (
        ("input_tokens", "output_tokens") if responses else ("prompt_tokens", "completion_tokens")
    )
    prompt = field(usage, input_name, 0) or 0
    completion = field(usage, output_name, 0) or 0
    total = field(usage, "total_tokens")
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total if total is not None else prompt + completion,
    }
    for detail, names in (
        (field(usage, f"{input_name}_details"), ("cached_tokens", "cache_write_tokens")),
        (field(usage, f"{output_name}_details"), ("reasoning_tokens",)),
    ):
        for name in names:
            value = field(detail, name)
            if value is not None:
                result[name] = value
    return result
