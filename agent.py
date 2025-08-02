import os
import asyncio
import logging
import requests
import socket
import time
import re
from typing import Optional, Dict, Any, List

import sounddevice as sd
import numpy as np
from bs4 import BeautifulSoup

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai as lk_openai
from livekit.plugins import noise_cancellation

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("agent_error.log", mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("realtime_agent")

# -----------------------------------------------------------------------------
# Env loading
# -----------------------------------------------------------------------------
load_dotenv()

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
def check_internet(timeout=5) -> bool:
    try:
        logger.info("Checking internet connectivity (DNS + HTTP)...")
        socket.gethostbyname("www.google.com")
        requests.get("https://www.google.com", timeout=timeout)
        logger.info("Internet connectivity OK.")
        return True
    except Exception as e:
        logger.error(f"Internet connectivity check failed: {e}")
        return False

def check_openai_api(key: str) -> bool:
    try:
        logger.info("Checking OpenAI API reachability...")
        resp = requests.get("https://api.openai.com/v1/models",
                            headers={"Authorization": f"Bearer {key}"}, timeout=10)
        if resp.status_code == 200:
            logger.info("OpenAI API reachable and key accepted.")
            return True
        else:
            logger.error(f"OpenAI model list error {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"OpenAI reachability check failed: {e}")
        return False

def check_microphone(sample_ms=1000) -> bool:
    try:
        logger.info("Checking microphone availability and basic input level...")
        devices = sd.query_devices()
        input_devs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if not input_devs:
            logger.error("No input devices detected.")
            return False
        logger.info(f"Input devices found: {[d['name'] for d in input_devs]}")
        def cb(indata, frames, time_info, status):
            vol = float(np.linalg.norm(indata)) * 10
            logger.debug(f"Mic instantaneous volume: {vol:.2f}")
        with sd.InputStream(callback=cb):
            sd.sleep(sample_ms)
        logger.info("Microphone basic test passed.")
        return True
    except Exception as e:
        logger.error(f"Microphone check failed: {e}")
        return False

# -----------------------------------------------------------------------------
# Simple tools
# -----------------------------------------------------------------------------
def google_search(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    Very basic web search using DuckDuckGo HTML (to avoid custom API keys).
    For production, use a proper API (Google Custom Search / SerpAPI).
    """
    logger.info(f"Tool: google_search('{query}', num_results={num_results})")
    try:
        url = "https://duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RealTimeAgent/1.0)"
        }
        res = requests.post(url, data=params, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        results = []
        for a in soup.select(".result__a")[:num_results]:
            title = a.get_text(strip=True)
            href = a.get("href")
            results.append({"title": title, "url": href})
        logger.debug(f"google_search results: {results}")
        return results
    except Exception as e:
        logger.exception(f"google_search failed: {e}")
        return []

def fetch_page_text(url: str, max_chars: int = 2000) -> str:
    logger.info(f"Tool: fetch_page_text('{url}')")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RealTimeAgent/1.0)"}
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]
    except Exception as e:
        logger.exception(f"fetch_page_text failed: {e}")
        return ""

# -----------------------------------------------------------------------------
# Assistant Agent
# -----------------------------------------------------------------------------
AGENT_INSTRUCTIONS = """You are a real-time assistant. You can:
- Answer general questions using your model knowledge.
- When asked to 'search' or 'look up', use the google_search tool and fetch_page_text to gather info and summarize.
- Keep responses concise unless the user requests detail.
- If the user says 'voice reply' or 'text reply', switch the mode accordingly.
"""

SESSION_GREETING = "Hello! I’m online. How can I help you? You can say 'voice reply' or 'text reply' at any time."

class RealTimeAssistant(Agent):
    def __init__(self, tool_fns):
        super().__init__(
            instructions=AGENT_INSTRUCTIONS,
            tools=tool_fns
        )
        self.reply_mode = "voice"  # "voice" or "text"

    async def on_user_message(self, ctx, message: str):
        logger.info(f"User said: {message}")
        # Mode switching by phrases
        lower = message.lower()
        if "voice reply" in lower:
            self.reply_mode = "voice"
            await ctx.send_message("Okay, I will reply in voice.")
            logger.info("Switched reply mode to voice.")
            return
        if "text reply" in lower:
            self.reply_mode = "text"
            await ctx.send_message("Okay, I will reply in text.")
            logger.info("Switched reply mode to text.")
            return

        await super().on_user_message(ctx, message)

# -----------------------------------------------------------------------------
# Reliable wrapper for generating a reply
# -----------------------------------------------------------------------------
async def safe_generate_reply(session: AgentSession, instructions: str, retries: int = 3, delay: float = 2.0):
    for i in range(retries):
        try:
            await session.generate_reply(instructions=instructions)
            return
        except Exception as e:
            logger.warning(f"generate_reply attempt {i+1}/{retries} failed: {e}")
            if i < retries - 1:
                await asyncio.sleep(delay)
            else:
                logger.error("All generate_reply attempts failed.")
                raise

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    # Read env
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    livekit_url = os.getenv("LIVEKIT_URL", "")
    livekit_api_key = os.getenv("LIVEKIT_API_KEY", "")
    livekit_api_secret = os.getenv("LIVEKIT_API_SECRET", "")
    voice = os.getenv("ASSISTANT_VOICE", "alloy")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    realtime_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")

    # Basic validations
    missing = []
    if not openai_api_key: missing.append("OPENAI_API_KEY")
    if not livekit_url: missing.append("LIVEKIT_URL")
    if not livekit_api_key: missing.append("LIVEKIT_API_KEY")
    if not livekit_api_secret: missing.append("LIVEKIT_API_SECRET")
    if missing:
        logger.error(f"Missing env variables: {missing}")
        raise EnvironmentError(f"Missing env variables: {missing}")

    logger.debug(f"Using OpenAI key: {openai_api_key[:4]}...{openai_api_key[-4:]}")
    logger.debug(f"LiveKit URL: {livekit_url}")
    logger.debug(f"Voice: {voice}")
    logger.debug(f"Realtime model: {realtime_model}")
    logger.debug(f"Temperature: {temperature}")

    # Diagnostics
    if not check_internet():
        logger.error("Startup aborted: no internet.")
        return
    if not check_openai_api(openai_api_key):
        logger.error("Startup aborted: OpenAI API not reachable or key invalid.")
        return
    if not check_microphone():
        logger.warning("Microphone test failed. You may still proceed if text-only is acceptable.")

    # Configure OpenAI plugin defaults via env
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Configure model
    try:
        llm = lk_openai.realtime.RealtimeModel(
            model=realtime_model,
            voice=voice,
            temperature=temperature,
        )
        logger.info("OpenAI Realtime model configured.")
    except Exception as e:
        logger.exception(f"Failed to configure Realtime model: {e}")
        raise

    # Room input options
    try:
        room_opts = RoomInputOptions(
            video_enabled=os.getenv("VIDEO_ENABLED", "False") == "True",
            audio_enabled=os.getenv("AUDIO_ENABLED", "True") == "True",
            noise_cancellation=noise_cancellation.BVC() if os.getenv("USE_NOISE_CANCELLATION", "False") == "True" else None,
        )
        logger.info(f"RoomInputOptions configured: video_enabled={room_opts.video_enabled}, audio_enabled={room_opts.audio_enabled}, noise_cancellation={'on' if room_opts.noise_cancellation else 'off'}")
    except TypeError as e:
        logger.exception("RoomInputOptions signature mismatch. Check livekit-agents version.")
        raise

    # Define tools that Agent can call by name
    async def tool_google_search(context, query: str, num_results: int = 3) -> Any:
        logger.info(f"Tool invoked: google_search | query={query}")
        return google_search(query, num_results=num_results)

    async def tool_fetch_page_text(context, url: str) -> Any:
        logger.info(f"Tool invoked: fetch_page_text | url={url}")
        return {"url": url, "text": fetch_page_text(url)}

    # Create the Agent
    agent = RealTimeAssistant(
        tool_fns=[tool_google_search, tool_fetch_page_text]
    )

    # Start session
    try:
        session = AgentSession(llm=llm)
        logger.debug("AgentSession created.")
        await session.start(
            room=ctx.room,
            agent=agent,
            room_input_options=room_opts,
        )
        logger.info("Connected to LiveKit room and session started.")
        await ctx.connect()
        logger.info("Job context connected.")

        # Initial greeting
        await safe_generate_reply(session, SESSION_GREETING)

        # Keep the session alive and log status
        # The AgentSession handles real-time events; we just keep the task running.
        while True:
            await asyncio.sleep(5)
            logger.debug("Health: session alive, awaiting audio/text input.")
    except Exception as e:
        logger.exception(f"Critical failure during session: {e}")
        raise
    finally:
        try:
            await session.aclose()
            logger.debug("Session closed.")
        except Exception:
            pass

# -----------------------------------------------------------------------------
# CLI runner
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Extra verbose logging for the plugin
    logging.getLogger("livekit.plugins.openai").setLevel(logging.DEBUG)
    logging.getLogger("livekit.agents").setLevel(logging.DEBUG)

    # Run the worker
    try:
        agents.cli.run_app(
            agents.WorkerOptions(entrypoint_fnc=entrypoint)
        )
    except Exception as e:
        logger.exception(f"Worker crashed: {e}")

