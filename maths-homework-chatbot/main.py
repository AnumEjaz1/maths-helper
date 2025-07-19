import os
from dotenv import load_dotenv
import chainlit as cl
from agents import Agent, Runner
from agents.run import RunConfig
from agents import OpenAIChatCompletionsModel, AsyncOpenAI

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in .env file")


@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

    )

    model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",  
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("chat_history", [])
    cl.user_session.set("config", config)

    agent = Agent(
        name="Math Assistant",
        instructions="You are a helpful math assistant. Answer only math-related questions like algebra, arithmetic, geometry, and word problems. If the question is not related to math, politely say you can only answer math questions.",
        model=model
    )

    cl.user_session.set("agent", agent)

    await cl.Message(content="👋 Hi! I'm your Math Helper. Ask me any math question.").send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="🤔 Thinking...")
    await msg.send()

    agent = cl.user_session.get("agent")
    config = cl.user_session.get("config")
    history = cl.user_session.get("chat_history") or []

 
    history.append({"role": "user", "content": message.content})

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        result = Runner.run_sync(starting_agent = agent,
                    input=history,
                    run_config=config)
        
        response_content = result.final_output
        
        msg.content = response_content
        await msg.update()
    
        
        cl.user_session.set("chat_history", result.to_input_list())
        
        # Optional: Log the interaction
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")
        
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
