from openai import OpenAI


def run_chat_session():
    client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
    # Start a chat session
    chat_history = []

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Append the user message to the chat history
        chat_history.append({"role": "user", "content": user_input})

        # Call the model with the chat history
        chat_session = client.chat.completions.create(
            model="llama3",
            messages=chat_history,
            temperature=0.7,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        # Append the assistant's response to the chat history
        chat_history.append(chat_session.choices[0].message)

        try:
            chat_complition = client.chat.completions.create(messages=chat_history, model="llama3")
            model_responce = chat_complition.choices[0].message['content']
            print(f"AI-Assistant: {model_responce}")
            chat_history.append({"role": "assistant", "content": model_responce})
        except Exception as e:
            print(f"Error appending message: {e}")
            break
if __name__ == "__main__":
    run_chat_session()