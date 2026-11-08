#!/usr/bin/env python3
"""Lightweight zero-dependency asyncio MQTT 3.1.1 broker for local development & testing.

Supports CONNECT, CONNACK, PUBLISH (QoS 0/1), SUBSCRIBE (with wildcard # and +),
SUBACK, PUBACK, PINGREQ, PINGRESP, and DISCONNECT.
"""

from __future__ import annotations

import asyncio
import os
import signal
import struct
import sys
from typing import Dict, List, Set, Tuple


def _decode_variable_length(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """Decode MQTT remaining length field from bytes starting at offset.

    Returns (length, new_offset).
    """
    multiplier = 1
    value = 0
    while True:
        if offset >= len(data):
            raise ValueError("Incomplete variable length field")
        byte = data[offset]
        offset += 1
        value += (byte & 127) * multiplier
        multiplier *= 128
        if (byte & 128) == 0:
            break
    return value, offset


def _encode_variable_length(length: int) -> bytes:
    """Encode integer length into MQTT variable length bytes."""
    out = bytearray()
    while True:
        byte = length % 128
        length //= 128
        if length > 0:
            byte |= 128
        out.append(byte)
        if length == 0:
            break
    return bytes(out)


def _topic_matches(subscription: str, topic: str) -> bool:
    """Check if topic matches an MQTT subscription pattern (# and + wildcards)."""
    if subscription == "#" or subscription == topic:
        return True

    sub_parts = subscription.split("/")
    top_parts = topic.split("/")

    for i, sub_p in enumerate(sub_parts):
        if sub_p == "#":
            return True
        if i >= len(top_parts):
            return False
        if sub_p != "+" and sub_p != top_parts[i]:
            return False

    return len(sub_parts) == len(top_parts)


class ClientSession:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, client_id: str):
        self.reader = reader
        self.writer = writer
        self.client_id = client_id
        self.subscriptions: Set[str] = set()
        self.send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.running = True

    async def send_packet(self, packet: bytes) -> None:
        if self.running:
            await self.send_queue.put(packet)


class SimpleMQTTBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 1883):
        self.host = host
        self.port = port
        self.clients: Dict[str, ClientSession] = {}
        self.server: asyncio.Server | None = None

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
        addrs = ", ".join(str(sock.getsockname()) for sock in self.server.sockets)
        print(f"====================================================================")
        print(f"  BMO Simple MQTT Broker running on {addrs}")
        print(f"  Ready for local testing with tools/mqtt_client.py or main.py")
        print(f"====================================================================")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        session: ClientSession | None = None
        writer_task: asyncio.Task | None = None

        try:
            # 1. First packet must be CONNECT (0x10)
            header_byte = await reader.readexactly(1)
            pkt_type = header_byte[0] >> 4
            if pkt_type != 1:  # CONNECT
                writer.close()
                await writer.wait_closed()
                return

            rem_len = await self._read_rem_len(reader)
            body = await reader.readexactly(rem_len)

            # Parse CONNECT body
            # Protocol Name
            proto_len = struct.unpack("!H", body[0:2])[0]
            offset = 2 + proto_len
            proto_level = body[offset]
            connect_flags = body[offset + 1]
            keep_alive = struct.unpack("!H", body[offset + 2 : offset + 4])[0]
            offset += 4

            # Client ID
            client_id_len = struct.unpack("!H", body[offset : offset + 2])[0]
            offset += 2
            client_id = body[offset : offset + client_id_len].decode("utf-8", errors="replace")
            if not client_id:
                client_id = f"client_{peer[1]}"

            session = ClientSession(reader, writer, client_id)
            self.clients[client_id] = session

            # Start background writer task
            writer_task = asyncio.create_task(self._client_writer_loop(session))

            # Send CONNACK (0x20, rem_len=2, flags=0x00, return_code=0x00)
            connack = bytes([0x20, 0x02, 0x00, 0x00])
            await session.send_packet(connack)

            # 2. Main incoming packet loop
            while session.running:
                try:
                    hdr = await reader.readexactly(1)
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    break

                b0 = hdr[0]
                p_type = b0 >> 4
                p_flags = b0 & 0x0F

                r_len = await self._read_rem_len(reader)
                p_body = await reader.readexactly(r_len) if r_len > 0 else b""

                if p_type == 3:  # PUBLISH
                    await self._handle_publish(session, p_flags, p_body)
                elif p_type == 8:  # SUBSCRIBE
                    await self._handle_subscribe(session, p_body)
                elif p_type == 10:  # UNSUBSCRIBE
                    await self._handle_unsubscribe(session, p_body)
                elif p_type == 12:  # PINGREQ
                    # Send PINGRESP (0xD0, 0x00)
                    await session.send_packet(bytes([0xD0, 0x00]))
                elif p_type == 14:  # DISCONNECT
                    break

        except Exception as exc:
            pass
        finally:
            if session:
                session.running = False
                if session.client_id in self.clients:
                    del self.clients[session.client_id]
            if writer_task:
                writer_task.cancel()
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _client_writer_loop(self, session: ClientSession) -> None:
        try:
            while session.running:
                packet = await session.send_queue.get()
                session.writer.write(packet)
                await session.writer.drain()
        except asyncio.CancelledError:
            pass
        except Exception:
            session.running = False

    async def _read_rem_len(self, reader: asyncio.StreamReader) -> int:
        multiplier = 1
        value = 0
        while True:
            b = (await reader.readexactly(1))[0]
            value += (b & 127) * multiplier
            multiplier *= 128
            if (b & 128) == 0:
                break
        return value

    async def _handle_subscribe(self, session: ClientSession, body: bytes) -> None:
        if len(body) < 2:
            return
        packet_id = struct.unpack("!H", body[0:2])[0]
        offset = 2
        granted_qoses = []

        while offset < len(body):
            top_len = struct.unpack("!H", body[offset : offset + 2])[0]
            offset += 2
            topic = body[offset : offset + top_len].decode("utf-8", errors="replace")
            offset += top_len
            req_qos = body[offset] if offset < len(body) else 0
            offset += 1
            session.subscriptions.add(topic)
            granted_qoses.append(min(req_qos, 1))

        # Send SUBACK (0x90)
        rem = bytes([packet_id >> 8, packet_id & 0xFF]) + bytes(granted_qoses)
        suback = bytes([0x90]) + _encode_variable_length(len(rem)) + rem
        await session.send_packet(suback)

    async def _handle_unsubscribe(self, session: ClientSession, body: bytes) -> None:
        if len(body) < 2:
            return
        packet_id = struct.unpack("!H", body[0:2])[0]
        offset = 2
        while offset < len(body):
            top_len = struct.unpack("!H", body[offset : offset + 2])[0]
            offset += 2
            topic = body[offset : offset + top_len].decode("utf-8", errors="replace")
            offset += top_len
            session.subscriptions.discard(topic)

        # Send UNSUBACK (0xB0, rem_len=2)
        unsuback = bytes([0xB0, 0x02, packet_id >> 8, packet_id & 0xFF])
        await session.send_packet(unsuback)

    async def _handle_publish(self, session: ClientSession, flags: int, body: bytes) -> None:
        qos = (flags >> 1) & 0x03
        top_len = struct.unpack("!H", body[0:2])[0]
        topic = body[2 : 2 + top_len].decode("utf-8", errors="replace")
        offset = 2 + top_len

        packet_id = 0
        if qos > 0:
            packet_id = struct.unpack("!H", body[offset : offset + 2])[0]
            offset += 2

        payload = body[offset:]

        # If QoS 1, acknowledge with PUBACK (0x40)
        if qos == 1:
            puback = bytes([0x40, 0x02, packet_id >> 8, packet_id & 0xFF])
            await session.send_packet(puback)

        # Forward to matching subscribers
        # Construct QoS 0 outbound packet
        top_bytes = topic.encode("utf-8")
        out_rem = struct.pack("!H", len(top_bytes)) + top_bytes + payload
        out_pkt = bytes([0x30]) + _encode_variable_length(len(out_rem)) + out_rem

        for client_id, sub_session in list(self.clients.items()):
            if not sub_session.running:
                continue
            for pattern in sub_session.subscriptions:
                if _topic_matches(pattern, topic):
                    await sub_session.send_packet(out_pkt)
                    break


async def main() -> None:
    host = os.environ.get("BMO_MQTT_HOST", "127.0.0.1")
    port = int(os.environ.get("BMO_MQTT_PORT", "1883"))

    broker = SimpleMQTTBroker(host=host, port=port)
    await broker.start()

    stop_event = asyncio.Event()

    def _sig_handler(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _sig_handler)
        except NotImplementedError:
            pass

    try:
        await stop_event.wait()
    finally:
        print("\nShutting down broker...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
