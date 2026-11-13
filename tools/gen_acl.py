"""Generate Mosquitto ACL file from bmo_contracts topics registry."""

from __future__ import annotations

import sys
from pathlib import Path

from bmo_contracts.topics import (
    TOPICS,
    Role,
)

ROLES: tuple[Role, ...] = ("cv", "brain", "face_ui")


def generate_acl_text() -> str:
    """Generate Mosquitto ACL content deterministically based on TOPICS."""
    blocks: list[str] = []
    for role in ROLES:
        write_topics: set[str] = set()
        read_topics: set[str] = set()

        for topic_name, spec in TOPICS.items():
            if role in spec.pub_roles:
                write_topics.add(topic_name)
            if role in spec.sub_roles:
                read_topics.add(topic_name)

        role_lines = [f"user {role}"]
        for topic in sorted(write_topics):
            role_lines.append(f"topic write {topic}")
        for topic in sorted(read_topics):
            role_lines.append(f"topic read {topic}")

        blocks.append("\n".join(role_lines))

    return "\n\n".join(blocks) + "\n"


def main() -> None:
    acl_text = generate_acl_text()
    if "--stdout" in sys.argv:
        sys.stdout.write(acl_text)
        return

    out_path = Path(__file__).resolve().parent.parent / "infra" / "acl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(acl_text, encoding="utf-8")
    sys.stdout.write(acl_text)


if __name__ == "__main__":
    main()
