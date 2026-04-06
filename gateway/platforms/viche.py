"""
Viche AI agent network platform adapter.

Connects Hermes to the Viche agent registry so other AI agents can delegate
tasks to Hermes and receive responses.

Protocol:
  1. Register via HTTP POST /registry/register → receive agent UUID
  2. Open Phoenix Channel WebSocket at /agent/websocket
  3. Join channel "agent:<uuid>" to receive inbound messages
  4. Dispatch via handle_message() → full Hermes pipeline
  5. Reply via send_message Phoenix event with type "result"

Environment variables:
  VICHE_ENABLED            Set to "true" to activate
  VICHE_REGISTRY_URL       Viche server base URL (default: http://localhost:4000)
  VICHE_AGENT_NAME         Display name in registry (default: "hermes")
  VICHE_CAPABILITIES       Comma-separated capabilities (default: "coding,research")
  VICHE_REGISTRY_TOKEN     Optional token for private registry scoping
  VICHE_ALLOWED_AGENTS     Comma-separated agent UUIDs to allow (optional)
  VICHE_ALLOW_ALL_AGENTS   Set "true" to allow any agent (default: true)
  VICHE_HOME_CHANNEL       Agent UUID for cron delivery target
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

# Token lock to prevent two profiles from using the same registry identity
try:
    from gateway.status import acquire_scoped_lock, release_scoped_lock

    _HAS_SCOPED_LOCK = True
except ImportError:
    _HAS_SCOPED_LOCK = False

logger = logging.getLogger(__name__)

# Phoenix heartbeat interval (seconds)
_HEARTBEAT_INTERVAL = 30
# Maximum reconnect attempts before giving up
_MAX_RECONNECT_ATTEMPTS = 20
# Base reconnect backoff (doubles each time, capped at 300s)
_RECONNECT_BASE = 5
_RECONNECT_CAP = 300


def check_viche_requirements() -> bool:
    """Check that runtime dependencies and env vars are in place.

    Returns True if the adapter can be started, False otherwise.
    """
    if not os.getenv("VICHE_ENABLED", "").lower() in ("true", "1", "yes"):
        return False
    try:
        import aiohttp  # noqa: F401
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


class VicheAdapter(BasePlatformAdapter):
    """
    Viche AI agent network ↔ Hermes gateway adapter.

    Each unique sending agent UUID gets its own Hermes session via
    chat_id format: "viche:<sender_agent_id>".

    Message flow:
      Inbound: Phoenix new_message → MessageEvent → handle_message() → AIAgent
      Outbound: send() parses chat_id → Phoenix send_message event
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.VICHE)

        self._registry_url: str = os.getenv(
            "VICHE_REGISTRY_URL", "http://localhost:4000"
        ).rstrip("/")
        self._agent_name: str = os.getenv("VICHE_AGENT_NAME", "hermes")
        self._capabilities: list[str] = [
            c.strip()
            for c in os.getenv("VICHE_CAPABILITIES", "coding,research").split(",")
            if c.strip()
        ]
        self._registry_tokens: list[str] = [
            t.strip()
            for t in os.getenv("VICHE_REGISTRY_TOKEN", "").split(",")
            if t.strip()
        ]
        self._allowed_agents: set[str] = {
            a.strip()
            for a in os.getenv("VICHE_ALLOWED_AGENTS", "").split(",")
            if a.strip()
        }
        self._allow_all: bool = (
            os.getenv("VICHE_ALLOW_ALL_AGENTS", "true").lower() in ("true", "1", "yes")
            or not self._allowed_agents
        )

        # State
        self._agent_id: Optional[str] = None
        self._ws = None  # websockets connection
        self._ref_counter: int = 0  # Phoenix message ref counter
        self._join_ref: Optional[str] = None  # Phoenix channel join reference
        self._ws_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_attempts: int = 0

        # Pending pushes: ref → Future (for tracking send_message acks)
        self._pending_pushes: Dict[str, asyncio.Future] = {}

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Register with Viche registry and open Phoenix Channel WebSocket."""
        # Acquire scoped lock to prevent duplicate connections across profiles
        if _HAS_SCOPED_LOCK:
            lock_scope = "viche"
            lock_identity = f"{self._registry_url}:{self._agent_name}"
            acquired, existing = acquire_scoped_lock(lock_scope, lock_identity)
            if not acquired:
                self._set_fatal_error(
                    "duplicate_connection",
                    f"Another Hermes profile is already connected to Viche at {self._registry_url}",
                    retryable=False,
                )
                return False

        # Step 1: Register with Viche registry (HTTP)
        agent_id = await self._register_with_retry()
        if not agent_id:
            return False

        self._agent_id = agent_id
        logger.info("[viche] Registered as agent %s", agent_id)

        # Step 2: Connect Phoenix Channel WebSocket
        try:
            connected = await self._connect_websocket()
            if not connected:
                return False
        except Exception as e:
            logger.error("[viche] WebSocket connect failed: %s", e)
            self._set_fatal_error("websocket_error", str(e), retryable=True)
            return False

        self._mark_connected()
        logger.info(
            "[viche] Connected to Viche network as '%s' (id=%s)",
            self._agent_name,
            agent_id,
        )
        return True

    async def disconnect(self) -> None:
        """Disconnect from Viche network."""
        self._running = False

        # Cancel background tasks
        for task in (self._ws_task, self._heartbeat_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Release scoped lock
        if _HAS_SCOPED_LOCK:
            lock_scope = "viche"
            lock_identity = f"{self._registry_url}:{self._agent_name}"
            release_scoped_lock(lock_scope, lock_identity)

        self._mark_disconnected()
        logger.info("[viche] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Send a response back to a Viche agent.

        chat_id format: "viche:<target_agent_id>"
        Sends a Phoenix Channel "send_message" event with type "result".
        """
        if not self._ws:
            return SendResult(
                success=False, error="Not connected to Viche WebSocket", retryable=True
            )

        # Parse chat_id: "viche:<agent_uuid>"
        parts = chat_id.split(":", 1)
        if len(parts) < 2 or parts[0] != "viche":
            return SendResult(
                success=False, error=f"Invalid Viche chat_id format: {chat_id!r}"
            )

        to_agent = parts[1]

        try:
            ref = self._next_ref()
            payload = {
                "to": to_agent,
                "type": "result",
                "body": content,
            }
            # Phoenix v2 frame: [join_ref, ref, topic, event, payload]
            frame = [
                self._join_ref,
                ref,
                f"agent:{self._agent_id}",
                "send_message",
                payload,
            ]
            await self._ws_send(json.dumps(frame))
            logger.debug(
                "[viche] Sent result to agent %s (%d chars)", to_agent[:8], len(content)
            )
            return SendResult(success=True, message_id=ref)
        except Exception as e:
            logger.error("[viche] send() failed: %s", e)
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Viche is AI-to-AI; typing indicators are not meaningful."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return chat info for a Viche agent session."""
        parts = chat_id.split(":", 1)
        agent_id = parts[1] if len(parts) > 1 else chat_id
        return {
            "name": f"Viche Agent ({agent_id[:8]}...)",
            "type": "dm",
            "chat_id": chat_id,
        }

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def _register(self) -> Optional[str]:
        """Register with Viche registry via HTTP POST. Returns agent UUID or None."""
        import aiohttp

        body: Dict[str, Any] = {"capabilities": self._capabilities}
        if self._agent_name:
            body["name"] = self._agent_name
        if self._registry_tokens:
            body["registries"] = self._registry_tokens

        url = f"{self._registry_url}/registry/register"
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                async with session.post(url, json=body) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        logger.error(
                            "[viche] Registration failed: %d %s",
                            resp.status,
                            text[:200],
                        )
                        return None
                    data = await resp.json()
                    return data.get("id")
        except aiohttp.ClientError as e:
            logger.error("[viche] Registration HTTP error: %s", e)
            return None

    async def _register_with_retry(self) -> Optional[str]:
        """Register with exponential backoff. Returns agent UUID or None after exhausting retries."""
        for attempt in range(1, 4):  # 3 attempts
            agent_id = await self._register()
            if agent_id:
                return agent_id
            if attempt < 3:
                wait = 2.0 * attempt
                logger.warning(
                    "[viche] Registration attempt %d failed, retrying in %.0fs...",
                    attempt,
                    wait,
                )
                await asyncio.sleep(wait)
        logger.error("[viche] Registration failed after 3 attempts")
        self._set_fatal_error(
            "registration_failed",
            f"Could not register with Viche registry at {self._registry_url} after 3 attempts",
            retryable=True,
        )
        return None

    # ------------------------------------------------------------------
    # WebSocket / Phoenix Channel
    # ------------------------------------------------------------------

    async def _connect_websocket(self) -> bool:
        """Open WebSocket and join Phoenix agent channel."""
        import websockets

        ws_url = self._registry_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        # Phoenix adds /websocket suffix to socket paths, and vsn=2.0.0 for protocol version
        ws_url = (
            f"{ws_url}/agent/websocket/websocket?agent_id={self._agent_id}&vsn=2.0.0"
        )

        logger.debug("[viche] Connecting WebSocket: %s", ws_url)
        try:
            self._ws = await websockets.connect(ws_url, ping_interval=None)
        except Exception as e:
            logger.error("[viche] WebSocket connection error: %s", e)
            self._set_fatal_error("websocket_connect_failed", str(e), retryable=True)
            return False

        # Join the agent channel
        joined = await self._join_channel(f"agent:{self._agent_id}")
        if not joined:
            return False

        # Optionally join registry channels (for private registry scoping)
        for token in self._registry_tokens:
            await self._join_channel(f"registry:{token}", optional=True)

        # Start message receive loop and heartbeat
        self._ws_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        return True

    async def _join_channel(self, topic: str, optional: bool = False) -> bool:
        """Send phx_join and wait for ok reply."""
        ref = self._next_ref()
        join_ref = self._next_ref()
        # Phoenix v2 frame: [join_ref, ref, topic, event, payload]
        frame = [join_ref, ref, topic, "phx_join", {}]

        try:
            await self._ws_send(json.dumps(frame))
            # Wait for join reply (up to 5 seconds)
            deadline = asyncio.get_event_loop().time() + 5.0
            while asyncio.get_event_loop().time() < deadline:
                raw = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                # msg format: [join_ref, ref, topic, event, payload]
                if len(msg) >= 4 and msg[2] == topic and msg[3] == "phx_reply":
                    payload = msg[4] if len(msg) > 4 else {}
                    if isinstance(payload, dict) and payload.get("status") == "ok":
                        # Store join_ref for subsequent messages on this channel
                        self._join_ref = join_ref
                        logger.debug(
                            "[viche] Joined channel: %s (join_ref=%s)", topic, join_ref
                        )
                        return True
                    else:
                        logger.warning(
                            "[viche] Channel join failed for %s: %s", topic, payload
                        )
                        if not optional:
                            self._set_fatal_error(
                                "channel_join_failed",
                                f"Could not join Phoenix channel {topic}: {payload}",
                                retryable=True,
                            )
                        return False
        except (asyncio.TimeoutError, Exception) as e:
            if not optional:
                logger.error("[viche] Channel join error for %s: %s", topic, e)
                self._set_fatal_error("channel_join_error", str(e), retryable=True)
            else:
                logger.debug("[viche] Optional channel %s join failed: %s", topic, e)
        return False

    async def _receive_loop(self) -> None:
        """Main loop: receive Phoenix messages and dispatch via handle_message()."""
        channel_topic = f"agent:{self._agent_id}"
        logger.debug("[viche] Receive loop started for channel %s", channel_topic)

        try:
            async for raw in self._ws:
                if not self._running:
                    break
                try:
                    await self._handle_ws_message(raw)
                except Exception as e:
                    logger.error("[viche] Message dispatch error: %s", e, exc_info=True)
        except Exception as e:
            if self._running:
                logger.warning("[viche] WebSocket disconnected: %s", e)
                asyncio.create_task(self._reconnect())

    async def _handle_ws_message(self, raw: str) -> None:
        """Parse a Phoenix frame and dispatch inbound task messages."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("[viche] Non-JSON message ignored: %s", raw[:80])
            return

        # Phoenix frame: [join_ref, ref, topic, event, payload]
        if not isinstance(msg, list) or len(msg) < 4:
            return

        _join_ref, _ref, topic, event = msg[0], msg[1], msg[2], msg[3]
        payload = msg[4] if len(msg) > 4 else {}

        # Only process events on our agent channel
        if topic != f"agent:{self._agent_id}":
            return

        # Ignore heartbeat replies and join acks
        if event in ("phx_reply", "phx_close", "phx_error", "phx_leave"):
            return

        if event == "new_message":
            await self._dispatch_new_message(payload)

    async def _dispatch_new_message(self, payload: Dict[str, Any]) -> None:
        """Transform a Viche message payload into a MessageEvent and inject."""
        msg_id = payload.get("id", "")
        msg_type = payload.get("type", "task")
        from_agent = payload.get("from", "")
        body = payload.get("body", "")

        if not from_agent:
            logger.warning("[viche] Received message with no 'from' field, ignoring")
            return

        # Authorization check
        if not self._is_agent_allowed(from_agent):
            logger.info(
                "[viche] Ignoring unauthorized message from agent %s", from_agent[:8]
            )
            return

        # Prevent echo (messages from ourselves)
        if from_agent == self._agent_id:
            logger.debug("[viche] Ignoring echo from own agent ID")
            return

        # Only process tasks and pings (not results — we send results, don't loop on them)
        if msg_type == "result":
            logger.debug(
                "[viche] Ignoring result message from %s (not a task)", from_agent[:8]
            )
            return

        # Prefix text so the AI knows its context
        display_type = msg_type.capitalize()
        text = f"[Viche:{display_type} from {from_agent}]\n{body}"

        # chat_id uses sender agent ID — one session per sending agent
        chat_id = f"viche:{from_agent}"

        source = self.build_source(
            chat_id=chat_id,
            chat_name=f"Viche Agent ({from_agent[:8]}...)",
            chat_type="dm",
            user_id=from_agent,
            user_name=from_agent,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=msg_id,
        )

        logger.info(
            "[viche] Inbound %s from agent %s (%d chars)",
            msg_type,
            from_agent[:8],
            len(body),
        )
        await self.handle_message(event)

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Send Phoenix heartbeat every 30s to keep the connection alive."""
        try:
            while self._running and self._ws:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
                if not self._running or not self._ws:
                    break
                try:
                    ref = self._next_ref()
                    frame = [None, ref, "phoenix", "heartbeat", {}]
                    await self._ws_send(json.dumps(frame))
                    logger.debug("[viche] Heartbeat sent")
                except Exception as e:
                    logger.warning("[viche] Heartbeat failed: %s", e)
                    break
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Reconnection
    # ------------------------------------------------------------------

    async def _reconnect(self) -> None:
        """Reconnect to Viche with exponential backoff."""
        if self._reconnect_attempts >= _MAX_RECONNECT_ATTEMPTS:
            logger.error(
                "[viche] Max reconnection attempts reached (%d), giving up",
                _MAX_RECONNECT_ATTEMPTS,
            )
            self._set_fatal_error(
                "max_reconnect_attempts",
                f"Failed to reconnect to Viche after {_MAX_RECONNECT_ATTEMPTS} attempts",
                retryable=True,
            )
            await self._notify_fatal_error()
            return

        self._reconnect_attempts += 1
        backoff = min(
            _RECONNECT_BASE * (2 ** (self._reconnect_attempts - 1)), _RECONNECT_CAP
        )
        jitter = random.uniform(0, backoff * 0.1)
        wait = backoff + jitter

        logger.info(
            "[viche] Reconnecting in %.1fs (attempt %d/%d)...",
            wait,
            self._reconnect_attempts,
            _MAX_RECONNECT_ATTEMPTS,
        )
        await asyncio.sleep(wait)

        # Re-register (agent ID may change after server restart)
        agent_id = await self._register_with_retry()
        if not agent_id:
            asyncio.create_task(self._reconnect())
            return

        self._agent_id = agent_id

        try:
            connected = await self._connect_websocket()
            if connected:
                self._reconnect_attempts = 0
                self._mark_connected()
                logger.info("[viche] Reconnected successfully as agent %s", agent_id)
            else:
                asyncio.create_task(self._reconnect())
        except Exception as e:
            logger.error("[viche] Reconnect failed: %s", e)
            asyncio.create_task(self._reconnect())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_ref(self) -> str:
        """Generate a unique Phoenix message reference string."""
        self._ref_counter += 1
        return str(self._ref_counter)

    async def _ws_send(self, data: str) -> None:
        """Send raw data over the WebSocket."""
        if self._ws:
            await self._ws.send(data)

    def _is_agent_allowed(self, agent_id: str) -> bool:
        """Check whether the sending agent is allowed to interact with us."""
        if self._allow_all:
            return True
        return agent_id in self._allowed_agents

    # Public accessor for tests
    @property
    def agent_id(self) -> Optional[str]:
        return self._agent_id
