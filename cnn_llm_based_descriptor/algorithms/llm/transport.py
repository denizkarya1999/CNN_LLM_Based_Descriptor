"""Loopback-only Ollama requests, endpoint checks and model downloads."""
import ipaddress
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from .model_store import ensure_project_server


def validate_endpoint(url):
    parsed = urlsplit(url)
    try:
        local = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        port = parsed.port
    except ValueError:
        local, port = False, None
    if (not local or parsed.scheme not in {"http", "https"} or parsed.username
            or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"} or (port is not None and port < 1)):
        raise ValueError("The description server must be a local Ollama URL, such as http://127.0.0.1:11434.")
    return url.rstrip("/")


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("The local description server must not redirect frame data.")


def request_json(endpoint, route, payload, timeout):
    ensure_project_server(validate_endpoint(endpoint))
    request = Request(validate_endpoint(endpoint) + route,
                      data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        # Frame data stays on the selected loopback service even with proxy env vars.
        with build_opener(ProxyHandler({}), NoRedirects()).open(request, timeout=timeout) as response:
            data = response.read(1_048_577)
        if len(data) > 1_048_576:
            raise ValueError("Description response exceeded the size limit.")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise ValueError("Invalid description server response.")
        if result.get("error"):
            raise RuntimeError(str(result["error"])[:500])
        return result
    except HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError("Vision model missing. Use Settings → Descriptions → Download model, then reopen the input.") from exc
        raise RuntimeError(f"Ollama returned HTTP {exc.code}. Check the local model service.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Could not reach Ollama or the description timed out. Start Ollama and check Settings → Descriptions.") from exc


def download_model(endpoint, model, progress, cancelled):
    """Stream local Ollama download progress without blocking Tk."""
    ensure_project_server(validate_endpoint(endpoint))
    request = Request(validate_endpoint(endpoint) + "/api/pull",
                      data=json.dumps({"model": model, "stream": True}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with build_opener(ProxyHandler({}), NoRedirects()).open(request, timeout=120) as response:
            while not cancelled.is_set():
                line = response.readline(65536)
                if not line:
                    break
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(str(data["error"])[:500])
                status = data.get("status", "Downloading…")
                if data.get("total"):
                    status += f" · {100 * data.get('completed', 0) / data['total']:.0f}%"
                progress(status)
                if data.get("status") == "success":
                    return
        raise RuntimeError("Download cancelled or incomplete. Download again to resume.")
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Could not download the model. Start local Ollama and check your connection.") from exc
