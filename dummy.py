import os
from huggingface_hub import InferenceClient
import json

# --- Step 1: Connect to the Hugging Face Serverless API ---
# You need a Hugging Face token with read access to the model.
# Set this as an environment variable named 'HF_TOKEN'.
# You can get a token from https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize the InferenceClient with the desired model.
# This client will be our "LLM brain."
client = InferenceClient(model="meta-llama/Llama-4-Scout-17B-16E-Instruct", token=HF_TOKEN)

# --- Step 2: Define the Agent’s System Prompt ---
# This prompt is crucial. It tells the agent about its capabilities,
# the tools it has access to, and the format it must follow for its actions.
SYSTEM_PROMPT = """Answer the following questions as best you can.
You have access to the following tools:

get_weather: Get the current weather in a given location

Use this format:

Question: the input question
Thought: reasoning step
Action:

```json
{ "action": "tool_name", "action_input": {...} }
```

Observation: tool output

Repeat this cycle as needed.
Always end with:

Thought: I now know the final answer
Final Answer: <your answer>"""

# --- Step 3: Define a Dummy Tool ---
# In a real-world scenario, this would be a function that
# performs an API call, database query, or other task.
# Here, it's a simple function that returns a hardcoded string.
def get_weather(location: str):
    """A dummy tool to simulate fetching weather data."""
    return f"The weather in {location} is sunny with low temperatures."

# --- Step 4: Run the Agent Loop Manually ---
# We will create a list of messages that represents the ongoing conversation.
# This is how the LLM "remembers" the context.
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]

print("--- Initial Messages ---")
for message in messages:
    print(f"{message['role']}: {message['content']}")

# First, we ask the LLM to think and generate an action.
# We use the 'stop' argument to prevent the LLM from hallucinating
# the 'Observation' part of the cycle.
output = client.chat.completions.create(
    messages=messages,
    max_tokens=150,
    stop=["Observation:"]
)
llm_response = output.choices[0].message.content

print("\n--- LLM's Response (before Observation) ---")
print(llm_response)

# --- Step 5: Execute the Action and Get a Real Observation ---
# We parse the LLM's response to extract the tool name and arguments.
try:
    # We look for the JSON block to parse the action.
    if "```json" in llm_response:
        json_str = llm_response.split("```json")[1].split("```")[0].strip()
        action_dict = json.loads(json_str)
        
        tool_name = action_dict["action"]
        tool_input = action_dict["action_input"]

        print(f"\n--- Framework executes the tool: {tool_name} with input {tool_input} ---")
        
        # We call the dummy tool and get a real observation.
        if tool_name == "get_weather":
            observation = get_weather(tool_input.get("location"))
        else:
            observation = "Error: Unknown tool."
    else:
        observation = "Error: LLM did not provide a valid action."
except Exception as e:
    observation = f"Error during action parsing or execution: {e}"

# --- Step 6: Feed the Observation back to the LLM ---
# We add the LLM's response and our real observation to the conversation.
messages.append({"role": "assistant", "content": llm_response + f"\nObservation: {observation}"})

# Now, we ask the LLM to continue the conversation, this time with the real
# data from the tool's output. It should provide the Final Answer.
final_output = client.chat.completions.create(
    messages=messages,
    stream=False,
    max_tokens=200,
)

print("\n--- Final Output from LLM ---")
print(final_output.choices[0].message.content)
