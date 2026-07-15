"""
Quick manual smoke test for the WebSocket chat endpoint -- not part of
the test suite, just a way to sanity-check the whole pipeline (auth ->
agent -> RAG -> response) end to end from the command line.

    python scripts/test_chat.py <access_token>
"""

import asyncio
import sys
import json
import websockets


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_chat.py <access_token>")
        return

    token = sys.argv[1]
    uri = f"ws://localhost:8000/api/v1/voice/stream/test-session-1?token={token}"

    async with websockets.connect(uri) as ws:
        print("Connected. Type a message (Hinglish/English/Hindi), or 'quit' to exit.\n")
        while True:
            msg = input("You: ").strip()
            if msg.lower() == "quit":
                break
            await ws.send(json.dumps({"text": msg}))
            response = await ws.recv()
            data = json.loads(response)
            print(f"Shrivox: {data['response']}")
            print(f"   (intent={data['intent']}, latency={data['latency_ms']}ms, flag={data['safety_flag']})\n")


if __name__ == "__main__":
    asyncio.run(main())