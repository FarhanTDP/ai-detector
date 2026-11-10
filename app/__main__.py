"""`python -m app` — start the server on all network interfaces (LAN-accessible)."""

import os

import uvicorn


def main() -> None:
    host = os.getenv("AIDET_HOST", "0.0.0.0")
    port = int(os.getenv("AIDET_PORT", "8000"))
    uvicorn.run("app.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
