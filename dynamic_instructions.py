from agents import Agent, Runner, AsyncOpenAI, set_default_openai_client, set_tracing_disabled, set_default_openai_api, RunContextWrapper
from dotenv import load_dotenv
import os,asyncio
from dataclasses import dataclass
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

@dataclass
class Hotel_info:
    name: str
    city: str
    price_per_night: float
    available_rooms: int
    

def dynamic_instructions(context: RunContextWrapper[Hotel_info], agent: Agent[Hotel_info]) -> str:
    return f"You are a hotel management agent capable of storing and retrieving details for multiple hotels. Maintain a dynamic database of hotels, including their name, city, price per night, and room availability. Based on the user's query, identify the context (e.g., specific hotel name, city, or request for booking) and provide relevant information or perform actions like booking a room. If the query is ambiguous, ask for clarification (e.g.,. For booking requests, check room availability, confirm the booking, and update the available room count. Ensure responses are accurate, concise, and tailored to the user's request."


async def main():
  hotel_assistant = Agent(
      name="Hotel Customer care",
      instructions=dynamic_instructions,
      model="gemini-2.0-flash",
  )
  

  user_input = "Can you book a room for me at Grand Plaza in karachi for 3 nights ?"
  result =await Runner.run(hotel_assistant, user_input)
  print(result.final_output)
asyncio.run(main())  
  

  