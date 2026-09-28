"""Read-only loopback viewer for frames captured by the owning Isaac process.

No simulation imports or capture work happens here. The caller atomically replaces
latest.jpg and live-state.json, calls start(), and closes the viewer in finally.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


_HTML = b'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Isaac P0 live camera</title>
<style>
body{max-width:960px;margin:32px auto;padding:0 16px;background:#111827;color:#e5e7eb;font:16px system-ui,sans-serif}
h1{font-size:24px}img{display:block;width:100%;height:auto;min-height:240px;object-fit:contain;background:#030712;border-radius:8px}
#status{padding:12px 0;color:#93c5fd}small{color:#9ca3af}
</style>
<h1>Isaac P0 live camera</h1>
<p id="status" role="status">Waiting for the first captured frame...</p>
<img id="camera" alt="Latest frame captured by Isaac Sim">
<p><small>Preview requests up to 30 times per second; actual fresh frames depend on simulation rendering and connection speed. The viewer closes when the experiment exits. Repeated images do not count as new simulation frames.</small></p>
<script>
const statusEl=document.getElementById('status'),camera=document.getElementById('camera');
let previousUrl=null;
async function refresh(){
  try{
    const [stateResponse,imageResponse]=await Promise.all([
      fetch('/state.json',{cache:'no-store'}),fetch('/frame.jpg',{cache:'no-store'})]);
    if(!stateResponse.ok || !imageResponse.ok)throw new Error('Waiting for the first captured frame...');
    const state=await stateResponse.json(),blob=await imageResponse.blob();
    const nextUrl=URL.createObjectURL(blob);
    camera.src=nextUrl;
    if(previousUrl)URL.revokeObjectURL(previousUrl);
    previousUrl=nextUrl;
    const time=Number(state.sim_time);
    statusEl.textContent=String(state.camera_id ?? 'camera')+' | Frame '+String(state.frame_id ?? '?')+' | simulation '+
      (Number.isFinite(time)?time.toFixed(3)+' s':'?')+' | '+String(state.phase ?? 'running');
  }catch(error){statusEl.textContent=error.message+' (The experiment may be starting or stopped.)';}
  setTimeout(refresh,33);
}
refresh();
</script></html>'''


class _LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = False


class LiveView:
    """Serve only three fixed resources on IPv4 loopback, within caller lifetime."""

    def __init__(self, output: Path, port: int = 18766):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError('port must be an integer in 0..65535')
        self.output = Path(output)
        self.port = port
        self._server = None
        self._thread = None
        self._root_fd = None
        self._lifecycle = threading.Lock()
        self._read_lock = threading.Lock()

    @property
    def url(self) -> str:
        return f'http://127.0.0.1:{self.port}/'

    def _read_file(self, name: str, limit: int) -> bytes:
        # Pin the directory and reject links, FIFOs and devices. Atomic writer
        # replacement is safe: an open reader sees one complete file version.
        with self._read_lock:
            if self._root_fd is None:
                raise OSError('viewer has closed')
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=self._root_fd)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                    raise ValueError('live resource is not a bounded regular file')
                with os.fdopen(fd, 'rb', closefd=False) as handle:
                    result = handle.read(limit + 1)
                if len(result) > limit:
                    raise ValueError('live resource exceeds size limit')
                return result
            finally:
                os.close(fd)

    def start(self) -> 'LiveView':
        with self._lifecycle:
            if self._server is not None:
                return self
            if self.output.is_symlink():
                raise ValueError('output directory must not be a symlink')
            # Existing workspace runs/ may be an intentional ancestor symlink.
            # Resolve that path, then pin its final directory with O_NOFOLLOW.
            directory = self.output.resolve(strict=True)
            root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            self._root_fd = root_fd
            viewer = self

            class Handler(BaseHTTPRequestHandler):
                server_version = 'IsaacP0LiveView'
                sys_version = ''

                def setup(self):
                    super().setup()
                    self.connection.settimeout(2.0)

                def log_message(self, format, *args):
                    return  # Do not turn browser polls into persistent logs.

                def do_HEAD(self):
                    self._respond(head=True)

                def do_GET(self):
                    self._respond(head=False)

                def _respond(self, head):
                    try:
                        parsed = urlsplit(self.path)
                        route = parsed.path if self.path.startswith('/') and not parsed.netloc else ''
                    except ValueError:
                        route = ''
                    status_code, content_type = 200, 'text/html; charset=utf-8'
                    if route == '/':
                        body = _HTML
                    elif route in ('/frame.jpg', '/state.json'):
                        try:
                            if route == '/frame.jpg':
                                body = viewer._read_file('latest.jpg', 8 * 1024 * 1024)
                                content_type = 'image/jpeg'
                            else:
                                body = viewer._read_file('live-state.json', 64 * 1024)
                                state = json.loads(body)
                                if not isinstance(state, dict):
                                    raise ValueError('live state must be a JSON object')
                                content_type = 'application/json; charset=utf-8'
                        except (OSError, ValueError, TypeError):
                            status_code, content_type = 503, 'application/json; charset=utf-8'
                            body = b'{"status":"warming_up","message":"No captured live resource is available yet."}'
                    else:
                        status_code, content_type = 404, 'text/plain; charset=utf-8'
                        body = b'Not found\n'
                    try:
                        self.send_response(status_code)
                        self.send_header('Content-Type', content_type)
                        self.send_header('Content-Length', str(len(body)))
                        self.send_header('Cache-Control', 'no-store, max-age=0')
                        self.send_header('X-Content-Type-Options', 'nosniff')
                        self.send_header('Content-Security-Policy', "default-src 'none'; img-src blob: 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
                        if status_code == 503:
                            self.send_header('Retry-After', '1')
                        self.end_headers()
                        if not head:
                            self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        return

            try:
                server = _LoopbackServer(('127.0.0.1', self.port), Handler)
                self.port = server.server_address[1]
                thread = threading.Thread(target=server.serve_forever,
                                          kwargs={'poll_interval': 0.05},
                                          name='isaac-p0-live-view', daemon=True)
                self._server, self._thread = server, thread
                thread.start()
            except Exception as error:
                if self._server is not None:
                    self._server.server_close()
                self._server, self._thread, self._root_fd = None, None, None
                os.close(root_fd)
                if isinstance(error, OSError):
                    raise RuntimeError(f'Cannot bind Isaac live viewer at {self.url}; port may already be in use. No existing process was changed.') from error
                raise
            return self

    def close(self) -> None:
        with self._lifecycle:
            if self._server is None:
                return
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=3)
            with self._read_lock:
                os.close(self._root_fd)
                self._root_fd = None
            self._server, self._thread = None, None
