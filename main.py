# Note: The openai-python library support for Azure OpenAI is in preview.
# Note: This code sample requires OpenAI Python library version 1.0.0 or higher.
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import time

models = {"gpt4": "jmgpt4", "gpt3.5-Turbo": "jm35turbo16"}

prompt = [{"role": "system", "content": "You are an AI assistant that helps people find information."},
          {"role": "user", "content": "Create a 200-word short story for an elf in the style of J.R.R. Tolkien."}]

matrix = [
    {"max_tokens": 16000, "model": "gpt3.5-Turbo", "stream": False},
    {"max_tokens": 650, "model": "gpt3.5-Turbo", "stream": False},
    {"max_tokens": 650, "model": "gpt3.5-Turbo", "stream": True},
    # {"max_tokens": 8000, "model": "gpt4", "stream": False},
    {"max_tokens": 650, "model": "gpt4", "stream": False},
    {"max_tokens": 650, "model": "gpt4", "stream": True},
]


def chat(message_text: str, max_tokens: int = 200, stream: bool = False, model: str = models["gpt3.5-Turbo"]):
    """
    Sends a chat message to the OpenAI chat API and prints the response with some timing information.

    Args:
        message_text (str): The text of the message to send.
        max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 200.
        stream (bool, optional): Whether to stream the response or not. Defaults to False.
        model (str, optional): The name of the model to use. Defaults to models["gpt3.5-Turbo"].

    Returns:
        None
    """
    start = time.time()
    first_message_time = 0
    completion = client.chat.completions.create(
        model=model,  # model = "deployment_name"
        messages=message_text,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=stream
    )

    if stream:
        tokens = 0
        for chunk in completion:
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta and choice.delta.content is not None:
                    if first_message_time == 0:
                        first_message_time = time.time()
                    print(choice.delta.content, end='', flush=True)
                    time.sleep(.01)
                tokens += 1
        print("\n")
        print(f"Total tokens received: {tokens}")
    else:
        print(completion.choices[0].message.content)
        print(completion.usage)
    end = time.time()
    print()
    print('*'*65)
    if (first_message_time):
        print(
            f"* First message received after {first_message_time-start} seconds".ljust(64)+"*")
    print(f"* Total time taken: {end-start} seconds".ljust(64)+"*")
    print('*'*65)
    print()


load_dotenv()

client = AzureOpenAI(
    azure_endpoint="https://jmaoai.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview"
)

for m in matrix:
    input("Press Enter to continue...")
    print('#'*80)
    print(
        f"# Test with stream={m['stream']} max_tokens={m['max_tokens']} and model {m['model']}/{models[m['model']]}".ljust(79)+"#")
    print('#'*80)
    chat(prompt, max_tokens=m['max_tokens'],
         model=models[m['model']], stream=m['stream'])

print("\nDone!\n")
