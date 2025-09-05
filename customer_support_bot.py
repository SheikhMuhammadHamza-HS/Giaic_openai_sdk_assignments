from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool,input_guardrail,GuardrailFunctionOutput,InputGuardrailTripwireTriggered,TResponseInputItem,RunContextWrapper,set_tracing_export_api_key
from dotenv import load_dotenv
import os
from dataclasses import dataclass
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))




external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)


@dataclass
class CustomerQuery:
    is_offensive: bool
    is_sentiment: str  # e.g., "positive", "neutral", "negative"
    is_complexity: str  # e.g., "simple", "complex"
    query: str


guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="""
    You are a guardrail agent that monitors user input for offensive or negative language. If detected, block the input or rephrase it to maintain a positive tone. Flag complex queries or negative sentiment for escalation to the HumanAgent. Ensure all interactions are professional and aligned with customer support standards.
    """,
    model=model,
    output_type=CustomerQuery,
)

@input_guardrail
async def customer_query_guardrail( 
    ctx: RunContextWrapper[CustomerQuery], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    print(f"Guardrail output: {result.final_output}")
    
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered= result.final_output.is_offensive or result.final_output.is_sentiment == "negative" or result.final_output.is_complexity == "complex",
    )

@function_tool(is_enabled=True)
def get_order_status(order_id: str) -> str:
    """Fetch the order status for a given order ID."""
    # Simulated order database
    order_database = {
        "12345": "Shipped",
        "67890": "Processing",
        "54321": "Delivered",
    }
    return order_database.get(order_id, "Order ID not found.")
  
# bot agent
bot_agent = Agent(
    name="Customer Support Bot",
    instructions= """
    You are a customer support bot that handles basic FAQs about products (e.g., features, pricing) and retrieves order statuses using the provided tools. Analyze user queries to answer autonomously or fetch order details. Detect complex queries or negative sentiment and escalate to a human agent. Maintain a friendly tone, avoid offensive language, and use context to provide accurate responses.
    """,
    model=model,
    tools=[get_order_status],
    )
    
# Human agent
human_agent = Agent(
    name="Human Agent",
    instructions="""
    ou are a human support agent for escalated queries. Handle complex customer issues or negative sentiment cases passed from the bot. Provide empathetic and detailed responses, resolving issues professionally. Use context from the bot's interaction to ensure continuity.
    """,
    handoff_description="Escalated to human agent for complex or sensitive issues.",
    model=model,
)

# Triage agent that routes between bot and human agents
Triage_agent = Agent(
    name="Assistant",
    instructions="""
    You are a triage agent who evaluates user queries for complexity and sentiment. Route simple FAQs and order status handoff to the BotAgent. Escalate complex queries or those with negative sentiment handoff to the HumanAgent. Ensure queries are processed efficiently and guardrails are applied to block or rephrase offensive input.
    """,
    model=model,
    handoffs=[bot_agent, human_agent],
    input_guardrails=[customer_query_guardrail],
    )
    
    #  testing the triage agent with guardrails
async def main():
  try:
      user_input = "Your service is terrible! I want to know why my order 67890 is delayed and when will it arrive?" # guardrail triggered bcoz its not offensive
      user_input = " i just want to know the status of my order 12345." # guardrail not triggered
      result = await Runner.run(Triage_agent, user_input)
      print("Guardrail didn't trip - this is unexpected\n\n")
      print(result.final_output)
  except InputGuardrailTripwireTriggered:
        print(" guardrail tripped wire triggered")    

import asyncio
asyncio.run(main())
    
