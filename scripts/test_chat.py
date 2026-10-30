"""
Manual smoke test for the chat WebSocket -- not part of the test suite, just a
way to sanity-check the whole pipeline (auth -> ticket -> agent -> RAG ->
response) end to end from the command line.

    SHRIVOX_EMAIL=admin@example.com python scripts/test_chat.py [base_url]

The password is read from SHRIVOX_PASSWORD or prompted for (never passed on the
command line). Flow, mirroring what a real integration does:

  1. POST /api/v1/auth/login        -> dashboard token (kept in memory only)
  2. POST /api/v1/auth/chat-ticket  -> short-lived ticket bound to a new session
  3. open the WebSocket with the TICKET

The dashboard token is never sent to the WebSocket, and no credential is
printed. The ticket still travels in the URL query string (a WebSocket
limitation); it expires within CHAT_TICKET_TTL_SECONDS.
"""

import asyncio
import getpass
import json
import os
import sys

import httpx
import websockets

DEFAULT_BASE_URL = "http://localhost:8000"


async def fetch_chat_ticket(client: httpx.AsyncClient, email: str, password: str) -> dict:
    """Login, then exchange the dashboard token for a chat ticket.

    Returns {"ticket", "session_id", "expires_in", "ws_path"}.
    """
    login = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    login.raise_for_status()
    dashboard_token = login.json()["access_token"]

    issued = await client.post(
        "/api/v1/auth/chat-ticket", headers={"Authorization": f"Bearer {dashboard_token}"}
    )
    issued.raise_for_status()
    return issued.json()


def websocket_url(base_url: str, issued: dict) -> str:
    ws_base = base_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    return f"{ws_base}{issued['ws_path']}?token={issued['ticket']}"


async def main():
    base_url = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL).rstrip("/")
    email = os.environ.get("SHRIVOX_EMAIL") or input("Email: ").strip()
    password = os.environ.get("SHRIVOX_PASSWORD") or getpass.getpass("Password: ")

    async with httpx.AsyncClient(base_url=base_url, timeout=15) as client:
        try:
            issued = await fetch_chat_ticket(client, email, password)
        except httpx.HTTPStatusError as exc:
            print(f"Could not obtain a chat ticket: HTTP {exc.response.status_code}")
            return

    print(f"Got a chat ticket (valid {issued['expires_in']}s). Connecting...")
    async with websockets.connect(websocket_url(base_url, issued)) as ws:
        print("Connected. Type a message (Hinglish/English/Hindi), or 'quit' to exit.\n")
        while True:
            msg = input("You: ").strip()
            if msg.lower() == "quit":
                break
            await ws.send(json.dumps({"text": msg}))
            data = json.loads(await ws.recv())
            print(f"Shrivox: {data['response']}")
            print(f"   (intent={data['intent']}, latency={data['latency_ms']}ms, flag={data['safety_flag']})\n")


if __name__ == "__main__":
    asyncio.run(main())
