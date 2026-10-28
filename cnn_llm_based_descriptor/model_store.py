"""Run the project's local Ollama service with weights under Models/."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIRECTORY = ROOT / 'Models' / 'ollama'
PROJECT_ENDPOINT = 'http://127.0.0.1:11435'
_LOCK = threading.Lock()


def _available():
    try:
        with build_opener(ProxyHandler({})).open(PROJECT_ENDPOINT + '/api/version', timeout=.5) as response:
            return isinstance(json.loads(response.read(4096)).get('version'), str)
    except (OSError, ValueError, AttributeError):
        return False


def ensure_project_server(endpoint):
    """Start on demand; custom loopback endpoints remain user-managed."""
    if endpoint.rstrip('/') != PROJECT_ENDPOINT:
        return
    with _LOCK:
        runtime = ROOT / '.runtime'
        state_file = runtime / 'model_server.json'
        if _available():
            try:
                state = json.loads(state_file.read_text())
                if state.get('models') == str(MODEL_DIRECTORY):
                    return
            except (OSError, ValueError):
                pass
            raise RuntimeError('The project model port 11435 is already in use by another service. Choose its local URL in Settings or free that port.')
        binary = shutil.which('ollama')
        if not binary:
            raise RuntimeError('Install Ollama using Get Ollama in Settings → Descriptions, then reopen the input.')
        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        runtime.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment.update(OLLAMA_MODELS=str(MODEL_DIRECTORY), OLLAMA_HOST='127.0.0.1:11435',
                           OLLAMA_NO_CLOUD='1', OLLAMA_KEEP_ALIVE='5m')
        with (runtime / 'ollama.log').open('a') as log:
            process = subprocess.Popen([binary, 'serve'], env=environment, cwd=ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name != 'nt'))
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError('The local model service could not start. Check .runtime/ollama.log or the selected port.')
            if _available():
                state_file.write_text(json.dumps({'pid': process.pid, 'models': str(MODEL_DIRECTORY)}))
                return
            time.sleep(.1)
        process.terminate()
        try:
            process.wait(5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        raise RuntimeError('The local model service did not start in time. Reopen the input to try again.')
