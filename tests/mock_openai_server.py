"""Deterministic OpenAI-compatible endpoint for the Docker E2E profile only."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib handler API
        if self.path != "/health":
            self.send_error(404)
            return
        self._write_json(200, {"ok": True})

    def do_POST(self):  # noqa: N802 - stdlib handler API
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if body.get("model") != "e2e-mock":
            self._write_json(400, {"error": {"message": "unexpected test model"}})
            return
        prompt = "\n".join(
            str(message.get("content", ""))
            for message in body.get("messages", [])
            if isinstance(message, dict)
        )
        choice = "not-a-candidate" if "__mock_unknown_candidate__" in prompt else "billing"
        content = json.dumps({
            "intent": {
                "choice": choice,
                "confidence": 0.91,
                "probabilities": {choice: 0.91},
            }
        })
        self._write_json(200, {
            "id": "chatcmpl-e2e",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "e2e-mock"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 16, "total_tokens": 36},
        })

    def _write_json(self, status: int, value: dict):
        data = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A003 - stdlib handler API
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
