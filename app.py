import os
import chainlit as cl
from literalai import LiteralClient
from dotenv import load_dotenv
from decouple import config
import asyncio
import nest_asyncio
import websockets

from v1o6 import start_chat_v1o6
from v1o7 import start_chat_v1o7

load_dotenv()

nest_asyncio.apply() # Chainlit won't run without this (with FastAPI)

literal_client = LiteralClient(api_key=os.environ.get("LITERAL_API_KEY"))

async def notify_active_agent(agent_name):
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        await websocket.send(agent_name)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.set_chat_profiles
async def set_chat_profile():
    return [
        cl.ChatProfile(
            name="Financial Assistant 2.0",
            markdown_description="Financial Assistant 2.0",
        ),
        cl.ChatProfile(
            name="Financial Assistant 1.0",
            markdown_description="Financial Assistant 1.0",
        ),
    ]

@cl.on_chat_start
async def on_chat_start():
    chat_profile = cl.user_session.get("chat_profile")
    await cl.Message(
        content="Please type your investment thesis to get started.",
        author='Financial Assistant'
    ).send()

@cl.on_message
async def on_message(message):
    chat_profile = cl.user_session.get("chat_profile")
    message_content = message.content
    if chat_profile == "Financial Assistant 1.0":
        start_chat_v1o6(message_content)
    elif chat_profile == "Financial Assistant 2.0":
        start_chat_v1o7(message_content)
