#!/usr/bin/env python3
"""Lightweight asyncio MQTT broker supporting basic auth, Mosquitto ACLs, and retained messages."""

from __future__ import annotations

import asyncio
import os
import struct
from pathlib import Path


def topic_matches(sub: str, topic: str) -> bool:
    """Check if topic matches an MQTT subscription pattern (# and + wildcards)."""
    if sub in ("#", topic):
        return True
    sub_parts, top_parts = sub.split("/"), topic.split("/")
    for i, sp in enumerate(sub_parts):
        if sp == "#":
            return True
        if i >= len(top_parts) or (sp != "+" and sp != top_parts[i]):
            return False
    return len(sub_parts) == len(top_parts)


def parse_acl_file(path: str | Path | None) -> dict[str, dict[str, set[str]]]:
    """Parse Mosquitto ACL file mapping user -> {'read': set, 'write': set}."""
    acl: dict[str, dict[str, set[str]]] = {}
    p = Path(path) if path else None
    if not p or not p.is_file():
        return acl
    u = ""
    for parts in (line.strip().split() for line in p.read_text("utf-8").splitlines()):
        if len(parts) >= 2 and parts[0] == "user":
            u = parts[1]
            acl.setdefault(u, {"read": set(), "write": set()})
        elif len(parts) >= 3 and parts[0] == "topic" and u:
            for act in ("read", "write"):
                if parts[1] in (act, "readwrite"):
                    acl[u][act].add(parts[2])
    return acl


def parse_passwords_file(path: str | Path | None) -> dict[str, str]:
    """Parse username:password file."""
    p = Path(path) if path else None
    lines = p.read_text("utf-8").splitlines() if p and p.is_file() else []
    return dict(line.strip().split(":", 1) for line in lines if ":" in line and not line.startswith("#"))


def _str(b: bytes, off: int) -> tuple[str, int]:
    end = off + 2 + struct.unpack("!H", b[off : off + 2])[0]
    return b[off + 2 : end].decode("utf-8", "replace"), end


def _decode_varlen(data: bytes, off: int = 0) -> tuple[int, int]:
    m, val = 1, 0
    while off < len(data):
        b, off = data[off], off + 1
        val += (b & 127) * m
        m *= 128
        if not (b & 128):
            return val, off
    raise ValueError("Incomplete varlen")


def _encode_varlen(n: int) -> bytes:
    out = bytearray()
    while True:
        out.append((n % 128) | (128 if n >= 128 else 0))
        n //= 128
        if n == 0:
            return bytes(out)


class ClientSession:
    def __init__(self, r: asyncio.StreamReader, w: asyncio.StreamWriter, cid: str, user: str = ""):
        self.reader, self.writer, self.client_id, self.username = r, w, cid, user
        self.subscriptions: set[str] = set()
        self.send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.running = True

    async def send(self, pkt: bytes) -> None:
        if self.running:
            await self.send_queue.put(pkt)

    async def writer_loop(self) -> None:
        try:
            while self.running:
                self.writer.write(await self.send_queue.get())
                await self.writer.drain()
        except (asyncio.CancelledError, Exception):
            self.running = False


class SimpleMQTTBroker:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1883,
        require_auth: bool = False,
        users: dict[str, str] | None = None,
        acl_path: str | Path | None = None,
    ):
        self.host, self.port, self.require_auth, self.users = host, port, require_auth, users or {}
        self.acl = parse_acl_file(acl_path) if acl_path else None
        self.clients: dict[str, ClientSession] = {}
        self.server: asyncio.AbstractServer | None = None
        self.retained: dict[str, bytes] = {}

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def stop(self) -> None:
        for s in list(self.clients.values()):
            s.running = False
            s.writer.close()
        self.clients.clear()
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        session, writer_task = None, None
        try:
            hdr = await reader.readexactly(1)
            if (hdr[0] >> 4) != 1:
                return
            body = await reader.readexactly(await self._read_rem_len(reader))
            proto_len = struct.unpack("!H", body[:2])[0]
            flags, off = body[3 + proto_len], 6 + proto_len
            if body[2 + proto_len] >= 5:
                plen, off = _decode_varlen(body, off)
                off += plen
            cid, off = _str(body, off)
            cid = cid or f"client_{writer.get_extra_info('peername')[1]}"
            if flags & 0x04:
                if body[2 + proto_len] >= 5:
                    wlen, off = _decode_varlen(body, off)
                    off += wlen
                off = _str(body, _str(body, off)[1])[1]
            u = _str(body, off)[0] if (flags & 0x80) else ""
            off += (2 + struct.unpack("!H", body[off : off + 2])[0]) if (flags & 0x80) else 0
            p = _str(body, off)[0] if (flags & 0x40) else ""
            if self.require_auth and (not u or self.users.get(u) != p):
                writer.write(bytes([0x20, 0x02, 0x00, 0x05]))
                await writer.drain()
                return

            session = ClientSession(reader, writer, cid, u)
            self.clients[cid] = session
            writer_task = asyncio.create_task(session.writer_loop())
            await session.send(bytes([0x20, 0x02, 0x00, 0x00]))
            while session.running:
                try:
                    hdr = await reader.readexactly(1)
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    break
                p_type, p_flags = hdr[0] >> 4, hdr[0] & 0x0F
                body = await reader.readexactly(await self._read_rem_len(reader))
                if p_type == 3:
                    await self._handle_publish(session, p_flags, body)
                elif p_type == 8:
                    await self._handle_subscribe(session, body)
                elif p_type == 10:
                    pkt_id, off = struct.unpack("!H", body[:2])[0], 2
                    while off < len(body):
                        top, off = _str(body, off)
                        session.subscriptions.discard(top)
                    await session.send(bytes([0xB0, 0x02, pkt_id >> 8, pkt_id & 0xFF]))
                elif p_type == 12:
                    await session.send(bytes([0xD0, 0x00]))
                elif p_type == 14:
                    break
        except Exception:
            pass
        finally:
            if session:
                session.running = False
                self.clients.pop(session.client_id, None)
            if writer_task:
                writer_task.cancel()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_rem_len(self, r: asyncio.StreamReader) -> int:
        m, val = 1, 0
        while True:
            b = (await r.readexactly(1))[0]
            val += (b & 127) * m
            m *= 128
            if not (b & 128):
                return val

    async def _handle_subscribe(self, session: ClientSession, body: bytes) -> None:
        pkt_id, off, granted = struct.unpack("!H", body[:2])[0], 2, []
        new_subs: list[str] = []
        while off < len(body):
            top, off = _str(body, off)
            req_qos, off = (body[off], off + 1) if off < len(body) else (0, off)
            ok = not self.acl or any(topic_matches(p, top) for p in self.acl.get(session.username, {}).get("read", ()))
            granted.append(min(req_qos, 1) if ok else 0x80)
            if ok:
                session.subscriptions.add(top)
                new_subs.append(top)
        rem = bytes([pkt_id >> 8, pkt_id & 0xFF]) + bytes(granted)
        await session.send(bytes([0x90]) + _encode_varlen(len(rem)) + rem)
        sent_retained: set[str] = set()
        for top in new_subs:
            for rtopic, rpayload in self.retained.items():
                if rtopic not in sent_retained and topic_matches(top, rtopic):
                    sent_retained.add(rtopic)
                    rtopic_b = rtopic.encode("utf-8")
                    t_hdr = struct.pack("!H", len(rtopic_b)) + rtopic_b
                    pkt = bytes([0x31]) + _encode_varlen(len(t_hdr) + len(rpayload)) + t_hdr + rpayload
                    await session.send(pkt)

    async def _handle_publish(self, session: ClientSession, flags: int, body: bytes) -> None:
        qos, (topic, off) = (flags >> 1) & 0x03, _str(body, 0)
        pkt_id = struct.unpack("!H", body[off : off + 2])[0] if qos > 0 else 0
        payload = body[off + (2 if qos > 0 else 0) :]
        if qos == 1:
            await session.send(bytes([0x40, 0x02, pkt_id >> 8, pkt_id & 0xFF]))
        if self.acl:
            allowed = self.acl.get(session.username, {}).get("write", ())
            if not any(topic_matches(p, topic) for p in allowed):
                return
        if flags & 0x01:
            if payload:
                self.retained[topic] = payload
            else:
                self.retained.pop(topic, None)
        out = bytes([0x30]) + _encode_varlen(len(body[:off] + payload)) + body[:off] + payload
        for s in list(self.clients.values()):
            if s.running and any(topic_matches(p, topic) for p in s.subscriptions):
                await s.send(out)


async def main() -> None:
    broker = SimpleMQTTBroker(
        os.getenv("BMO_MQTT_HOST", "127.0.0.1"),
        int(os.getenv("BMO_MQTT_PORT", "1883")),
        os.getenv("BMO_MQTT_REQUIRE_AUTH", "0") in ("1", "true"),
        parse_passwords_file(os.getenv("BMO_MQTT_PASSWORDS_PATH")),
        os.getenv("BMO_MQTT_ACL_PATH"),
    )
    await broker.start()
    try:
        await asyncio.Event().wait()
    finally:
        await broker.stop()


if __name__ == "__main__":
    asyncio.run(main())
