import httpx
from openai import OpenAI

# Bypass http_proxy for loopback connections (Intel proxy returns 403 for localhost)
_transport = httpx.HTTPTransport(proxy=None)
client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="dummy",
    http_client=httpx.Client(transport=_transport),
)
model = "/data/Qwen3-235B-A22B-Thinking-2507-FP8"

questions = [
    "1+1=?",
    "please introduce the Capital City of China",
    "write a python application for bubble sort",
]

for q in questions:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": q}],
        max_tokens=256,
        temperature=0,
    )
    print(f"Q: {q}")
    print(f"A: {resp.choices[0].message.content}\n")