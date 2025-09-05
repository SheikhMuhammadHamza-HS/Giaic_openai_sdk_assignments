import os
import asyncio
from openai import AsyncOpenAI
from pydantic import BaseModel
from agents import Agent, Runner, input_guardrail, InputGuardrailTripwireTriggered, GuardrailFunctionOutput, TResponseInputItem, RunContextWrapper, set_tracing_disabled, set_default_openai_api, set_default_openai_client,output_guardrail,OutputGuardrailTripwireTriggered
from dotenv import load_dotenv
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(True)
set_default_openai_api("chat_completions")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
set_default_openai_client(external_client)

# input guardrail to check if the input is related to math homework
class MathHomeworkOutput(BaseModel):
    is_math_homework: bool
    reason: str
    answer: str 


# output guardrail to check the output from agent
class MessageOutput(BaseModel): 
    response: str

# output guardrail agent to classify if output is political content  
output_guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check If the response is about to political content ,So stop the execution",
    output_type=MessageOutput,
    model="gemini-2.0-flash",
)

# Define the guardrail agent to classify if input is math homework 
guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="Check if the user is asking about math homework related queries.",
    model="gemini-2.0-flash",
    output_type=MathHomeworkOutput
)

# 
@output_guardrail
async def math_guardrail(  
    ctx: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrail_agent, output.response, context=ctx.context)
    print(f"Guardrail output: {result.final_output}")
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered= not result.final_output.is_math,
)

@input_guardrail
async def math_homework_guardrail(ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    print(f"Guardrail output: {result.final_output}")
    # Reverse logic: Trigger guardrail if NOT math homework
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_math_homework,  
    )

agent = Agent(
    name="math expert agent",
    instructions="You are an expert in mathematics. Answer only math homework related questions.",
    input_guardrails=[math_homework_guardrail],
    output_guardrails=[math_guardrail],
    output_type=MessageOutput,
    model="gemini-2.0-flash",
)

async def main():
    try:
        await Runner.run(agent, "who is the founder of pakistan?")
        print("Successfully completed without triggering the guardrail.")
    except InputGuardrailTripwireTriggered as e:
        print("Guardrail was triggered. Stopping execution.")
        print(f"Guardrail output: {e}")
    except OutputGuardrailTripwireTriggered as e:
        print("Guardrail was triggered. Stopping execution.")
        print(f"Guardrail output: {e}")

asyncio.run(main())