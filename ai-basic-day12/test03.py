from openai import OpenAI
client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")

responce = client.chat.completions.create(
    model="llama3",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What is the capital of Germany?"},
    ],
    model="llama3",
    temperature=0.7,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

print(responce.choices[0].message['content'])
