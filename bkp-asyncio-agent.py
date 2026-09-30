import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import (
    McpToolset,
    StdioConnectionParams,
    StdioServerParameters,
)
from google.genai import types

# 1. Load keys from .env
load_dotenv()

MAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if not MAPS_KEY:
  raise ValueError("Missing GOOGLE_MAPS_API_KEY in .env file!")

# 2. Define root_agent for Google ADK
root_agent = LlmAgent(
    name="google_maps_agent",
    model=Gemini(model="gemini-2.5-flash"),
    instruction=(
        "You are an expert travel assistant with access to Google Maps. "
        "Use your Google Maps tools to look up directions, travel times, "
        "and geographical information when asked by the user."
    ),
    tools=[
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=[
                        "-y",
                        "@modelcontextprotocol/server-google-maps",
                    ],
                    env={"GOOGLE_MAPS_API_KEY": MAPS_KEY},
                ),
                timeout=30,
            ),
        )
    ],
)


# 3. Interactive Terminal Runner
async def main():
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="maps_app",
      agent=root_agent,
      session_service=session_service,
      auto_create_session=True,
  )
  session = await session_service.create_session(
      app_name="maps_app", user_id="user"
  )

  print("=" * 60)
  print(" Google Maps AI Agent (Powered by MCP)")
  print(" Type 'exit' to quit.")
  print("=" * 60)

  while True:
    prompt = input("\nYou: ").strip()
    if not prompt:
      continue
    if prompt.lower() in ["exit", "quit"]:
      print("Goodbye!")
      break

    print("\n[Calling Google Maps MCP server...]")
    content = types.Content(
        role="user", parts=[types.Part.from_text(text=prompt)]
    )

    try:
      async for event in runner.run_async(
          user_id="user", session_id=session.id, new_message=content
      ):
        # Print the final synthesized answer from the LLM
        if event.content and event.content.parts:
          for part in event.content.parts:
            if part.text:
              print(f"\nAgent:\n{part.text}")
    except Exception as e:
      print(f"\nError: {e}")


if __name__ == "__main__":
  asyncio.run(main())