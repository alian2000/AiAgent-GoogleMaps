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

# 2. Define the Agent with BOTH Google Maps and Weather MCP Servers
root_agent = LlmAgent(
    name="travel_assistant",
    model=Gemini(model="gemini-2.5-flash"),
    instruction=(
        "You are an expert travel assistant. "
        "Use Google Maps tools for directions, routes, and places. "
        "Use Weather tools for real-time weather forecasts, current conditions, and alerts. "
        "When asked questions involving travel, provide both directions and weather information."
    ),
    tools=[
        # Toolset 1: Google Maps MCP Server
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
        ),
        # Toolset 2: Free Weather MCP Server (No API Key Required!)
        McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command="npx",
                    args=[
                        "-y",
                        "@dangahagan/weather-mcp",
                    ],
                ),
                timeout=30,
            ),
        ),
    ],
)


# 3. Synchronous Terminal Runner
def main():
  session_service = InMemorySessionService()
  runner = Runner(
      app_name="maps_app",
      agent=root_agent,
      session_service=session_service,
      auto_create_session=True,
  )

  print("=" * 60)
  print(" Travel & Weather AI Agent (Maps + Weather MCP)")
  print(" Type 'exit' to quit.")
  print("=" * 60)

  while True:
    prompt = input("\nYou: ").strip()
    if not prompt:
      continue
    if prompt.lower() in ["exit", "quit"]:
      print("Goodbye!")
      break

    print("\n[Calling Maps & Weather MCP servers...]")
    content = types.Content(
        role="user", parts=[types.Part.from_text(text=prompt)]
    )

    try:
      for event in runner.run(
          user_id="user", session_id="my_session", new_message=content
      ):
        if event.content and event.content.parts:
          for part in event.content.parts:
            if part.text:
              print(f"\nAgent:\n{part.text}")
    except Exception as e:
      print(f"\nError: {e}")


if __name__ == "__main__":
  main()