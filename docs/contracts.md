# BMO Contracts Specification

This document defines the inter-process communication contracts for the BMO Assistant ecosystem across Computer Vision (`cv`), LLM & Dialogue Orchestrator (`brain`), Edge/Audio compute (`pi`), and Microcontrollers (`esp32`).

---

## 4.1 Topics Registry (`bmo_contracts/topics.py`)

All topics are statically declared in `bmo_contracts.topics`. String literals representing topics must never be declared outside of `bmo_contracts.topics`.

| Constant | Topic | Dir | QoS | Retain | Model | Pub Roles | Sub Roles |
|---|---|---|---|---|---|---|---|
| `VISION_STATUS` | `bmo/vision/status` | cv → all | 1 | yes (+LWT) | `VisionStatus` | `cv` | `brain`, `pi` |
| `VISION_EVENTS` | `bmo/vision/events` | cv → brain | 1 | no | `Event` | `cv` | `brain` |
| `VISION_CONTEXT` | `bmo/vision/context` | cv → brain | 0 | yes | `Scene` | `cv` | `brain` |
| `VISION_CMD` | `bmo/vision/cmd` | brain → cv | 1 | no | `VisionCmd` | `brain` | `cv` |
| `VISION_CMD_RESULT` | `bmo/vision/cmd_result` | cv → brain | 1 | no | `VisionCmdResult` | `cv` | `brain` |
| `BRAIN_STATUS` | `bmo/brain/status` | brain → all | 1 | yes (+LWT) | `BrainStatus` | `brain` | `cv`, `pi` |
| `BRAIN_SAY` | `bmo/brain/say` | brain → pi | 1 | no | `BrainSay` | `brain` | `pi` |
| `HW_CMD` | `bmo/hw/cmd` | brain → edge | 1 | no | `HwCmd` | `brain` | `pi`, `esp32` |
| `HW_STATUS` | `bmo/hw/status` | edge → all | 1 | yes (+LWT) | `HwStatus` | `pi`, `esp32` | `brain` |
| `HW_EVENT` | `bmo/hw/event` | edge → brain | 1 | no | `HwEvent` | `pi`, `esp32` | `brain` |

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

### 3. HwStatus (`bmo/hw/status`)
```json
{
  "online": true,
  "ts": 1700000000.0,
  "capabilities": ["display", "audio", "servo"]
}
```

### 4. HwEvent (`bmo/hw/event`)
```json
{
  "ts": 1700000000.123,
  "kind": "mute_switch",
  "value": true
}
```

### 5. FaceIdentified Event (`bmo/vision/events`)
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

### 6. PersonDetected Event (`bmo/vision/events`)
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

### 7. PersonLeft Event (`bmo/vision/events`)
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

### 8. SignRecognized Event (`bmo/vision/events`)
```json
{
  "v": 1,
  "id": "evt-000004",
  "ts": 1700000004.567,
  "type": "sign_recognized",
  "data": {
    "text": "hello",
    "conf": 0.92
  }
}
```

### 9. ObjectSeen Event (`bmo/vision/events`)
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

### 10. Scene Context (`bmo/vision/context`)
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
  "last_sign": {
    "text": "hello",
    "age_s": 3.2
  },
  "recent": [
    "person detected (track: 1)",
    "face identified: Alice",
    "sign: hello"
  ]
}
```

### 11. VisionCmd (`bmo/vision/cmd`)
```json
{
  "id": "cmd-001",
  "cmd": "reload_gallery",
  "args": {}
}
```

### 12. VisionCmdResult (`bmo/vision/cmd_result`)
```json
{
  "id": "cmd-001",
  "ok": true,
  "detail": "gallery reloaded successfully"
}
```

### 13. BrainSay (`bmo/brain/say`)
```json
{
  "utt_id": "utt-001",
  "text": "Hello there! How can I help you today?",
  "emotion": "happy"
}
```

### 14. HwCmd (`bmo/hw/cmd`)
```json
{
  "id": "hw-001",
  "target": "pi",
  "action": "set_expression",
  "args": {
    "expression": "happy"
  }
}
```
