# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# ...modifies code from GPT-5, existing code...
import json
import queue
import time
import logging
import threading
import uuid
from typing import Dict, Any, Generator

from flask import Blueprint, Response, session, request, jsonify, current_app, stream_with_context

logger = logging.getLogger(__name__)

sse_bp = Blueprint('sse', __name__, url_prefix='/api/sse')

# Thread-safe mapping: session_id -> { 'queue': Queue(), 'connected_clients': [connection_id,...] }
sse_connections_lock = threading.RLock()
sse_connections: Dict[str, Dict[str, Any]] = {}


def format_sse_message(data: Dict[str, Any]) -> str:
    """Format a message for SSE transmission"""
    if 'timestamp' not in data:
        data['timestamp'] = time.time()
    try:
        json_data = json.dumps(data)
    except Exception:
        json_data = json.dumps({"error": "serialization_failed", "orig": str(data)})
    return f"data: {json_data}\n\n"


def event_stream_from_queue(client_queue: "queue.Queue", session_id: str, connection_id: str) -> Generator[str, None, None]:
    """
    Generator that yields SSE frames from the provided queue.
    Sends periodic heartbeats and catches exceptions so the worker does not exit.
    """
    # initial handshake
    yield format_sse_message({"type": "connected", "connection_id": connection_id})

    last_heartbeat = time.time()
    try:
        while True:
            try:
                item = client_queue.get(timeout=1)
            except queue.Empty:
                # heartbeat comment to keep connection alive
                if time.time() - last_heartbeat > 30:
                    last_heartbeat = time.time()
                    yield format_sse_message({"type": "heartbeat", "ts": time.time()})
                else:
                    yield ": heartbeat\n\n"
                continue

            try:
                yield format_sse_message(item)
            except Exception as e:
                current_app.logger.exception("Error encoding SSE item for session %s connection %s", session_id, connection_id)
                yield format_sse_message({"type": "error", "message": str(e)})
    except SystemExit:
        current_app.logger.warning("Caught SystemExit in SSE generator for session %s connection %s", session_id, connection_id)
        return
    except Exception:
        current_app.logger.exception("Unhandled exception in SSE generator for session %s connection %s", session_id, connection_id)
        return


@sse_bp.route('/connect')
def sse_connect():
    """
    Establish an SSE connection for the client's session.
    Accepts session id from cookie 'session_id', header 'X-Session-Id', query param 'session_id', or Flask session.
    """
    session_id = (
        request.cookies.get("session_id")
        or request.headers.get("X-Session-Id")
        or request.args.get("session_id")
        or session.get("session_id")
    )

    if not session_id:
        current_app.logger.info("SSE connect rejected: no session id provided")
        return Response("No session ID found", status=401)

    connection_id = f"conn_{uuid.uuid4().hex[:8]}"
    current_app.logger.info("SSE connect: session=%s connection=%s", session_id, connection_id)

    # create or reuse session queue and register connection
    with sse_connections_lock:
        entry = sse_connections.get(session_id)
        if not entry:
            entry = {"queue": queue.Queue(), "connected_clients": []}
            sse_connections[session_id] = entry
        entry["connected_clients"].append(connection_id)
        client_queue = entry["queue"]
        current_app.logger.debug("SSE registered connection: %s -> %s", session_id, entry["connected_clients"])

    def generator():
        try:
            yield from event_stream_from_queue(client_queue, session_id, connection_id)
        finally:
            # cleanup: remove connection id and drop session entry if no clients left
            try:
                with sse_connections_lock:
                    entry = sse_connections.get(session_id)
                    if entry and connection_id in entry.get("connected_clients", []):
                        entry["connected_clients"].remove(connection_id)
                        current_app.logger.info("SSE connection closed: session=%s connection=%s", session_id, connection_id)
                        if not entry["connected_clients"]:
                            # clear queue to free memory
                            try:
                                while not entry["queue"].empty():
                                    entry["queue"].get_nowait()
                            except Exception:
                                pass
                            sse_connections.pop(session_id, None)
                            current_app.logger.info("SSE session removed: %s", session_id)
            except Exception:
                current_app.logger.exception("Error during SSE cleanup for session %s connection %s", session_id, connection_id)

    headers = {
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    }
    return Response(stream_with_context(generator()), mimetype='text/event-stream', headers=headers)


@sse_bp.route('/send-message', methods=['POST'])
def send_message_to_session():
    """
    POST JSON {"session_id": "...", "payload": {...}} to push a message to all connected clients for that session.
    """
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400

    target_session_id = body.get('session_id')
    payload = body.get('payload', {"text": "ping"})

    if not target_session_id:
        return jsonify({"error": "session_id is required"}), 400

    with sse_connections_lock:
        entry = sse_connections.get(target_session_id)
        if not entry:
            return jsonify({"error": "session not found or not connected"}), 404
        try:
            entry["queue"].put_nowait(payload)
        except queue.Full:
            current_app.logger.warning("SSE queue full for session %s", target_session_id)
            return jsonify({"error": "queue full"}), 503

    return jsonify({"status": "Message queued"}), 200


@sse_bp.route('/broadcast', methods=['POST'])
def broadcast_message():
    """
    Broadcast a message to all connected SSE clients
    Expected JSON payload: {"message": {...}}
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "invalid json"}), 400

    message = data.get('message', {})
    sent_count = 0
    with sse_connections_lock:
        session_ids = list(sse_connections.keys())

    for sid in session_ids:
        with sse_connections_lock:
            entry = sse_connections.get(sid)
            if entry:
                try:
                    entry["queue"].put_nowait(message)
                    sent_count += 1
                except queue.Full:
                    current_app.logger.warning("SSE queue full for session %s", sid)

    current_app.logger.info("Broadcasted message to %d sessions", sent_count)
    return jsonify({"status": f"Message broadcasted to {sent_count} sessions"}), 200


@sse_bp.route('/sse-connection-check', methods=['POST'])
def sse_connection_check():
    session_id = request.json.get('session_id')
    if session_id in sse_connections:
        return jsonify({"status": "connected"})
    else:
        return jsonify({"status": "disconnected"}), 404


@sse_bp.route('/status', endpoint='sse_status')
def get_sse_status():
    """Get the current status of SSE connections"""
    connection_info = {}
    with sse_connections_lock:
        for sid, conn in sse_connections.items():
            connection_info[sid] = {
                'connected_clients': list(conn.get('connected_clients', [])),
                'queue_size': conn.get('queue').qsize() if conn.get('queue') else 0
            }
    return jsonify({"connected_sessions": connection_info}), 200