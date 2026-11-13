# BMO Contracts Specification

This document defines the inter-process communication contracts for the BMO Assistant ecosystem across Computer Vision (`cv`) and LLM & Dialogue Orchestrator (`brain`).

---

## 4.1 Topics Registry (`bmo_contracts/topics.py`)

All topics are statically declared in `bmo_contracts.topics`. String literals representing topics must never be declared outside of `bmo_contracts.topics`.

| Constant | Topic | Dir | QoS | Retain | Model | Pub Roles | Sub Roles |
|---|---|---|---|---|---|---|---|
| `VISION_STATUS` | `bmo/vision/status` | cv → all | 1 | yes (+LWT) | `VisionStatus` | `cv` | `brain` |
| `VISION_EVENTS` | `bmo/vision/events` | cv → brain | 1 | no | `Event` | `cv` | `brain` |
| `VISION_CONTEXT` | `bmo/vision/context` | cv → brain | 0 | yes | `Scene` | `cv` | `brain` |
| `VISION_CMD` | `bmo/vision/cmd` | brain → cv | 1 | no | `VisionCmd` | `brain` | `cv` |
| `VISION_CMD_RESULT` | `bmo/vision/cmd_result` | cv → brain | 1 | no | `VisionCmdResult` | `cv` | `brain` |
| `BRAIN_STATUS` | `bmo/brain/status` | brain → all | 1 | yes (+LWT) | `BrainStatus` | `brain` | `cv` |
| `BRAIN_SAY` | `bmo/brain/say` | brain → local TTS | 1 | no | `BrainSay` | `brain` | |

---

## 4.2 Models (`bmo_contracts/models.py`)

All contract payloads are validated using Pydantic v2 with strict schema enforcement (`ConfigDict(extra="forbid")`).

---

## 4.3 Verbatim JSON Payload Examples

### 1. VisionStatus (`bmo/vision/status`)
```json
{
  "online": true
}
```

### 2. BrainStatus (`bmo/brain/status`)
```json
{
  "online": true,
  "state": "listening"
}
```

### 3. FaceIdentified Event (`bmo/vision/events`)
```json
{
  "v": 1,
  "id": "evt-000001",
  "ts": 1700000001.234,
  "type": "face_identified",
  "data": {
    "track_id": 1,
    "name": "Alice",
    "conf": 0.95
  }
}
```

### 4. PersonDetected Event (`bmo/vision/events`)
```json
{
  "v": 1,
  "id": "evt-000002",
  "ts": 1700000002.345,
  "type": "person_detected",
  "data": {
    "track_id": 1,
    "pos": "center",
    "proximity": "near"
  }
}
```

### 5. PersonLeft Event (`bmo/vision/events`)
```json
{
  "v": 1,
  "id": "evt-000003",
  "ts": 1700000003.456,
  "type": "person_left",
  "data": {
    "track_id": 1
  }
}
```

### 6. ObjectSeen Event (`bmo/vision/events`)
```json
{
  "v": 1,
  "id": "evt-000005",
  "ts": 1700000005.678,
  "type": "object_seen",
  "data": {
    "label": "cup",
    "conf": 0.88
  }
}
```

### 7. Scene Context (`bmo/vision/context`)
```json
{
  "ts": 1700000006.789,
  "people": [
    {
      "track_id": 1,
      "name": "Alice",
      "pos": "center",
      "proximity": "near",
      "since_s": 12.5
    }
  ],
  "objects": ["cup", "bottle"],
  "recent": [
    "person detected (track: 1)",
    "face identified: Alice"
  ]
}
```

### 8. VisionCmd (`bmo/vision/cmd`)
```json
{
  "id": "cmd-001",
  "cmd": "reload_gallery",
  "args": {}
}
```

### 9. VisionCmdResult (`bmo/vision/cmd_result`)
```json
{
  "id": "cmd-001",
  "ok": true,
  "detail": "gallery reloaded successfully"
}
```

### 10. BrainSay (`bmo/brain/say`)
```json
{
  "utt_id": "utt-001",
  "text": "Hello there! How can I help you today?",
  "emotion": "happy"
}
```