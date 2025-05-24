# %% [markdown]
# # Semantic Kernel 
# 
# In this code sample, you will use the [Semantic Kernel](https://aka.ms/ai-agents-beginners/semantic-kernel) AI Framework to create a basic agent. 
# 
# The goal of this sample is to show you the steps that we will later use in the additional code samples when implementing the different agentic patterns. 

# %% [markdown]
# ## Import the Needed Python Packages 

# %%
import os 
from typing import Annotated
from openai import AsyncOpenAI

from dotenv import load_dotenv

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

# %% [markdown]
# ## Creating the Client
# 
# In this sample, we will use [GitHub Models](https://aka.ms/ai-agents-beginners/github-models) for access to the LLM. 
# 
# The `ai_model_id` is defined as `gpt-4o-mini`. Try changing the model to another model available on the GitHub Models marketplace to see the different results. 
# 
# For us to use the `Azure Inference SDK` that is used for the `base_url` for GitHub Models, we will use the `OpenAIChatCompletion` connector within Semantic Kernel. There are also other [available connectors](https://learn.microsoft.com/semantic-kernel/concepts/ai-services/chat-completion) to use Semantic Kernel for other model providers.

# %%
import random   

# Define a sample plugin for the sample

class DestinationsPlugin:
    """A List of Random Destinations for a vacation."""

    def __init__(self):
        # List of vacation destinations
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia"
        ]
        # Track last destination to avoid repeats
        self.last_destination = None

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(self) -> Annotated[str, "Returns a random vacation destination."]:
        # Get available destinations (excluding last one if possible)
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)

        # Select a random destination
        destination = random.choice(available_destinations)

        # Update the last destination
        self.last_destination = destination

        return destination

# %%
load_dotenv()

# Check if the API key is set correctly
api_key = os.environ.get("GITHUB_TOKEN")
if not api_key:
    raise ValueError("API key is not set. Please check your .env file.")

# Check if the API key has the required permissions
# Instead of checking the key directly, we can attempt to use it and catch errors
try:
    client = AsyncOpenAI(
        api_key=api_key, 
        base_url="https://models.inference.ai.azure.com/",
    )
    # Attempt to create the chat completion service to validate permissions
    chat_completion_service = OpenAIChatCompletion(
        ai_model_id="gpt-4o-mini",
        async_client=client,
    )
except AuthenticationError as e:
    print(f"Authentication error: {e}")
    raise ValueError("API key does not have the required 'models' permission. Please obtain a key with the necessary permissions.")
except Exception as e:
    print(f"Failed to create chat completion service: {e}")
    raise

# %% [markdown]
# ## Creating the Agent 
# 
# Below we create the Agent called `TravelAgent`.
# 
# For this example, we are using very simple instructions. You can change these instructions to see how the agent responds differently. 

# %%
agent = ChatCompletionAgent(
    service=chat_completion_service, 
    plugins=[DestinationsPlugin()],
    name="TravelAgent",
    instructions="You are a helpful AI Agent that can help plan vacations for customers at random destinations",
)

# %% [markdown]
# ## Running the Agent
# 
# Now we can run the Agent by defining a thread of type `ChatHistoryAgentThread`.  Any required system messages are provided to the agent's invoke_stream `messages` keyword argument.
# 
# After these are defined, we create a `user_inputs` that will be what the user is sending to the agent. In this case, we have set this message to `Plan me a sunny vacation`. 
# 
# Feel free to change this message to see how the agent responds differently. 

# %%
async def main():
    # Create a new thread for the agent
    # If no thread is provided, a new thread will be
    # created and returned with the initial response
    thread: ChatHistoryAgentThread | None = None

    user_inputs = [
        "Plan me a day trip.",
    ]

    for user_input in user_inputs:
        print(f"# User: {user_input}\n")
        first_chunk = True
        async for response in agent.invoke_stream(
            messages=user_input, thread=thread,
        ):
            # 5. Print the response
            if first_chunk:
                print(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            print(f"{response}", end="", flush=True)
            thread = response.thread
        print()

    # Clean up the thread
    await thread.delete() if thread else None

async def run():
    await main()

# Call the run function to execute the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(run())


