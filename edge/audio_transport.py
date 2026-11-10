"""UDP and HTTP audio transport for edge-brain streaming."""

from __future__ import annotations

import io
import queue
import socket
import struct
import threading
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from brain.audio.io import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    AudioSink,
    AudioSource,
    to_int16,
)


def pcm_to_wav_bytes(pcm: np.ndarray | bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode mono 16-bit PCM audio samples into WAV bytes."""
    raw = to_int16(pcm).tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)
    return buf.getvalue()


def wav_bytes_to_pcm(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes into a 1D int16 PCM numpy array."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return to_int16(raw)


class UDPAudioSender:
    """Sends 16 kHz mono int16 audio frames over UDP with sequence numbers."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    def send_frame(self, frame: np.ndarray | bytes) -> int:
        """Send a single frame padded or sliced to FRAME_SAMPLES."""
        arr = to_int16(frame)

        if len(arr) < FRAME_SAMPLES:
            arr = np.pad(arr, (0, FRAME_SAMPLES - len(arr)))
        elif len(arr) > FRAME_SAMPLES:
            arr = arr[:FRAME_SAMPLES]

        packet = struct.pack("!I", self._seq) + arr.tobytes()
        sent = self.sock.sendto(packet, (self.host, self.port))
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return sent

    def send_pcm(self, pcm: np.ndarray | bytes) -> int:
        """Split PCM samples into FRAME_SAMPLES frames and send all."""
        arr = to_int16(pcm)

        total = 0
        for i in range(0, len(arr), FRAME_SAMPLES):
            chunk = arr[i : i + FRAME_SAMPLES]
            total += self.send_frame(chunk)
        return total

    def close(self) -> None:
        """Close the underlying UDP socket."""
        self.sock.close()


class UDPAudioReceiver(AudioSource):
    """Receives 16 kHz mono int16 audio frames over UDP into a frame queue."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.bind((host, port))
        self.port: int = self.sock.getsockname()[1]
        self.sock.settimeout(0.2)

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._running = True
        self.received_packets = 0
        self.dropped_packets = 0
        self.late_packets = 0
        self._expected_seq: int | None = None

        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def _receive_loop(self) -> None:
        while self._running:
            try:
                data, _ = self.sock.recvfrom(2048)
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                break

            if len(data) < 4:
                continue

            seq = struct.unpack("!I", data[:4])[0]
            if self._expected_seq is None:
                self._expected_seq = (seq + 1) & 0xFFFFFFFF
            else:
                gap = (seq - self._expected_seq) & 0xFFFFFFFF
                if gap >= 0x80000000:
                    self.late_packets += 1
                    continue
                if gap > 0:
                    self.dropped_packets += gap
                    zero_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16)
                    fill_count = min(gap, 500)
                    for _ in range(fill_count):
                        self._queue.put(zero_frame.copy())
                self._expected_seq = (seq + 1) & 0xFFFFFFFF

            self.received_packets += 1
            raw_frame = data[4 : 4 + FRAME_BYTES]
            if len(raw_frame) < FRAME_BYTES:
                raw_frame = raw_frame.ljust(FRAME_BYTES, b"\x00")
            frame = to_int16(raw_frame)
            self._queue.put(frame)

    def read_frame(self, timeout_s: float | None = 0.2) -> np.ndarray | None:
        """Read the next received audio frame from the queue."""
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def close(self) -> None:
        """Stop background worker and close the UDP socket."""
        self._running = False
        try:
            self.sock.close()
        except OSError:
            pass
        if self._thread.is_alive() and self._thread != threading.current_thread():
            self._thread.join(timeout=1.0)


class _AudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        req_handler: type[BaseHTTPRequestHandler],
    ) -> None:
        super().__init__(server_address, req_handler)
        self.lock = threading.Lock()
        self.utterances: dict[str, bytes] = {}
        self.latest_wav: bytes | None = None

    def store_wav(self, utt_id: str, wav: bytes) -> None:
        with self.lock:
            self.utterances[utt_id] = wav
            self.latest_wav = wav

    def get_latest_wav(self) -> bytes | None:
        with self.lock:
            return self.latest_wav

    def get_utterance_wav(self, utt_id: str) -> bytes | None:
        with self.lock:
            return self.utterances.get(utt_id)


class _AudioHTTPRequestHandler(BaseHTTPRequestHandler):
    server: _AudioServer

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/tts/latest.wav", "/tts/latest"):
            payload = self.server.get_latest_wav()
        elif path.startswith("/tts/"):
            utt_id = path[len("/tts/") :]
            if utt_id.endswith(".wav"):
                utt_id = utt_id[:-4]
            payload = self.server.get_utterance_wav(utt_id)
        else:
            payload = None

        if payload is not None:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        _ = (format, args)


class HTTPAudioServer(AudioSink):
    """Serves TTS WAV replies over HTTP."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._httpd = _AudioServer((host, port), _AudioHTTPRequestHandler)
        self.host: str = host
        self.port: int = self._httpd.server_address[1]
        self.base_url: str = f"http://{host}:{self.port}"
        self._running = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def store_utterance(self, utt_id: str, pcm: np.ndarray | bytes) -> str:
        """Store PCM audio encoded as WAV by utt_id and update latest."""
        wav_bytes = pcm_to_wav_bytes(pcm)
        self._httpd.store_wav(utt_id, wav_bytes)
        return f"{self.base_url}/tts/{utt_id}.wav"

    def write_samples(self, pcm: np.ndarray | bytes) -> None:
        """Write PCM samples as the latest utterance."""
        self.store_utterance("latest", pcm)

    def stop(self) -> None:
        """Stop HTTP server and join thread."""
        if self._running:
            self._running = False
            self._httpd.shutdown()
            self._httpd.server_close()
            if self._thread.is_alive() and self._thread != threading.current_thread():
                self._thread.join(timeout=1.0)

    def close(self) -> None:
        """Close audio sink."""
        self.stop()


def fetch_tts_wav(url: str, timeout_s: float = 5.0) -> np.ndarray:
    """Fetch WAV over HTTP and return 1D int16 PCM array."""
    req = urllib.request.Request(url, headers={"User-Agent": "BMO-AudioTransport/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        wav_bytes = response.read()
    return wav_bytes_to_pcm(wav_bytes)
