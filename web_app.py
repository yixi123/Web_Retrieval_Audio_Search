import json
import os
import shutil
import tempfile
import uuid
from urllib.parse import unquote

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from flask import send_file
from werkzeug.utils import secure_filename

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from audio_indexing import (
    index_audio_file,
    semantic_search_audio,
    semantic_search_audio_by_file,
    list_indexed_files as get_indexed_files,
)


app = Flask(__name__, static_folder='static', template_folder='templates')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_ROOT = os.path.join(APP_ROOT, 'user_uploads')
os.makedirs(UPLOAD_ROOT, exist_ok=True)

GLOBAL_INDEX_NAME = 'audio_segments'
GLOBAL_METADATA_PATH = 'audio_metadata.json'
ALLOWED_EXT = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
SESSION_UPLOAD_LIMIT_BYTES = 500 * 1024 * 1024
TOTAL_USER_UPLOAD_LIMIT_BYTES = 10 * 1024 * 1024 * 1024
TEMP_DIR_PREFIXES = ('audio_query_', 'audio_user_query_')


def allowed(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT




def _user_paths(session_id: str):
    session_dir = os.path.join(UPLOAD_ROOT, session_id)
    index_name = os.path.join(session_dir, f'user_{session_id}')
    metadata_path = os.path.join(session_dir, f'user_{session_id}_metadata.json')
    return session_dir, index_name, metadata_path


def _folder_size(path: str):
    total = 0
    if not os.path.exists(path):
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _upload_size_bytes(uploaded_files):
    total = 0
    for uploaded in uploaded_files:
        stream = getattr(uploaded, 'stream', None)
        if stream is None:
            continue
        try:
            current_pos = stream.tell()
            stream.seek(0, os.SEEK_END)
            total += int(stream.tell())
            stream.seek(current_pos, os.SEEK_SET)
        except (OSError, AttributeError, ValueError):
            content_length = getattr(uploaded, 'content_length', None)
            if content_length:
                total += int(content_length)
    return total


def _user_session_dirs():
    for name in os.listdir(UPLOAD_ROOT):
        if name.startswith(TEMP_DIR_PREFIXES):
            continue
        session_dir = os.path.join(UPLOAD_ROOT, name)
        if os.path.isdir(session_dir):
            yield name, session_dir


def _current_user_upload_total_bytes():
    total = 0
    for _, session_dir in _user_session_dirs():
        total += _folder_size(session_dir)
    return total


def _prune_oldest_user_sessions(required_bytes: int):
    current_total = _current_user_upload_total_bytes()
    if current_total + required_bytes <= TOTAL_USER_UPLOAD_LIMIT_BYTES:
        return []

    sessions = []
    for session_id, session_dir in _user_session_dirs():
        try:
            mtime = os.path.getmtime(session_dir)
        except OSError:
            mtime = 0
        sessions.append((mtime, session_id, session_dir, _folder_size(session_dir)))

    sessions.sort(key=lambda item: item[0])
    removed_sessions = []

    for _, session_id, session_dir, size in sessions:
        if current_total + required_bytes <= TOTAL_USER_UPLOAD_LIMIT_BYTES:
            break
        shutil.rmtree(session_dir, ignore_errors=True)
        current_total = max(0, current_total - size)
        removed_sessions.append(session_id)

    return removed_sessions


def _save_files(uploaded_files, session_dir: str):
    saved = []
    skipped_duplicates = []
    os.makedirs(session_dir, exist_ok=True)
    existing_names = {
        name
        for name in os.listdir(session_dir)
        if os.path.isfile(os.path.join(session_dir, name))
    }
    for uploaded in uploaded_files:
        filename = secure_filename(uploaded.filename)
        if not filename or not allowed(filename):
            continue
        if filename in existing_names:
            skipped_duplicates.append(filename)
            continue
        destination = os.path.join(session_dir, filename)
        uploaded.save(destination)
        saved.append(destination)
        existing_names.add(filename)
    return saved, skipped_duplicates


def _safe_existing_path(file_path: str):
    file_path = os.path.normpath(unquote(file_path))
    candidate_paths = []
    if not os.path.isabs(file_path):
        candidate_paths.append(os.path.abspath(file_path))
    candidate_paths.append(file_path)

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/ping')
def ping():
    return jsonify({'status': 'ok'})


@app.route('/api/search/text', methods=['POST'])
def global_text_search():
    payload = request.get_json(force=True) or {}
    query_text = (payload.get('query') or '').strip()
    top_k = int(payload.get('top_k') or 50)
    if not query_text:
        return jsonify({'results': []})
    return jsonify({'results': semantic_search_audio(query_text, top_k=top_k, index_name=GLOBAL_INDEX_NAME, metadata_path=GLOBAL_METADATA_PATH)})


@app.route('/api/search/audio', methods=['POST'])
def global_audio_search():
    if 'audio' not in request.files:
        return jsonify({'results': [], 'error': 'missing audio'}), 400

    top_k = int(request.form.get('top_k') or 50)
    uploaded = request.files['audio']
    temp_dir = tempfile.mkdtemp(prefix='audio_query_', dir=UPLOAD_ROOT)
    temp_path = os.path.join(temp_dir, secure_filename(uploaded.filename or 'query.wav'))
    uploaded.save(temp_path)
    try:
        return jsonify({'results': semantic_search_audio_by_file(temp_path, top_k=top_k, index_name=GLOBAL_INDEX_NAME, metadata_path=GLOBAL_METADATA_PATH)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/user/upload', methods=['POST'])
def upload():
    session_id = request.form.get('session_id') or str(uuid.uuid4())
    session_dir, index_name, metadata_path = _user_paths(session_id)
    uploaded_files = []
    for uploaded in request.files.getlist('files'):
        filename = secure_filename(uploaded.filename)
        if not filename or not allowed(filename):
            continue
        uploaded_files.append(uploaded)

    if not uploaded_files:
        return jsonify({'session_id': session_id, 'files': [], 'error': 'no valid audio files'}), 400

    incoming_size = _upload_size_bytes(uploaded_files)
    existing_session_size = _folder_size(session_dir)
    if existing_session_size + incoming_size > SESSION_UPLOAD_LIMIT_BYTES:
        return jsonify({
            'session_id': session_id,
            'files': [],
            'error': 'session upload limit exceeded (500 MB). Please clear this session before uploading more files.',
        }), 413

    pruned_sessions = _prune_oldest_user_sessions(incoming_size)
    saved_paths, duplicate_files = _save_files(uploaded_files, session_dir)

    # Return saved files info but DON'T index them here
    # Let /api/user/index handle indexing and return the correct count
    saved_info = []
    for audio_path in saved_paths:
        saved_info.append({
            'file_id': str(uuid.uuid4()),
            'path': audio_path,
            'filename': os.path.basename(audio_path),
        })

    return jsonify({'session_id': session_id, 'files': saved_info, 'pruned_sessions': pruned_sessions, 'duplicate_files': duplicate_files})


@app.route('/api/user/index', methods=['POST'])
def index_user_session():
    data = request.get_json(force=True) or {}
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'missing session_id'}), 400

    session_dir, index_name, metadata_path = _user_paths(session_id)
    if not os.path.exists(session_dir):
        return jsonify({'error': 'session not found'}), 404

    already_indexed = {os.path.abspath(path) for path in get_indexed_files(metadata_path=metadata_path)}
    
    # FIX 1: Check if the key exists in the data payload, not if it's truthy
    if 'files_to_index' in data:
        audio_files = [
            os.path.join(session_dir, name)
            for name in data['files_to_index']
            if os.path.isfile(os.path.join(session_dir, name)) and allowed(name)
        ]
        audio_files = [path for path in audio_files if os.path.abspath(path) not in already_indexed]
    else:
        # Fallback: index all unindexed files in the session
        audio_files = [
            os.path.join(session_dir, name)
            for name in sorted(os.listdir(session_dir))
            if os.path.isfile(os.path.join(session_dir, name)) and allowed(name)
        ]
        audio_files = [path for path in audio_files if os.path.abspath(path) not in already_indexed]


    def generate():
        total = len(audio_files)
        yield json.dumps({'type': 'start', 'total': total}) + '\n'
        indexed_files = []
        
        for index, audio_path in enumerate(audio_files, start=1):
            # FIX 2: Capture the result to verify successful indexing
            result = index_audio_file(audio_path, index_name=index_name, metadata_path=metadata_path, verbose=False)
            
            if result is not None:
                indexed_files.append({
                    'file_id': str(uuid.uuid4()),
                    'path': audio_path,
                    'filename': os.path.basename(audio_path),
                })
                
            # Yield progress to keep the UI bar moving, even if the file failed
            yield json.dumps({
                'type': 'progress',
                'current': index,
                'total': total,
                'filename': os.path.basename(audio_path),
            }) + '\n'

        yield json.dumps({
            'type': 'done',
            'session_id': session_id,
            # FIX 3: Return the actual successful count, not the total attempted
            'indexed_count': len(indexed_files),
            'session_total_files': len([name for name in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, name)) and allowed(name)]),
            'files': indexed_files,
        }) + '\n'

    return Response(generate(), mimetype='application/x-ndjson')


@app.route('/api/user/search/text', methods=['POST'])
def user_text_search():
    payload = request.get_json(force=True) or {}
    session_id = payload.get('session_id')
    query_text = (payload.get('query') or '').strip()
    top_k = int(payload.get('top_k') or 50)
    if not session_id:
        return jsonify({'results': [], 'error': 'missing session_id'}), 400
    if not query_text:
        return jsonify({'results': []})

    _, index_name, metadata_path = _user_paths(session_id)
    return jsonify({'results': semantic_search_audio(query_text, top_k=top_k, index_name=index_name, metadata_path=metadata_path)})


@app.route('/api/user/search/audio', methods=['POST'])
def user_audio_search():
    session_id = request.form.get('session_id')
    if not session_id:
        return jsonify({'results': [], 'error': 'missing session_id'}), 400
    if 'audio' not in request.files:
        return jsonify({'results': [], 'error': 'missing audio'}), 400

    top_k = int(request.form.get('top_k') or 50)
    _, index_name, metadata_path = _user_paths(session_id)
    temp_dir = tempfile.mkdtemp(prefix='audio_user_query_', dir=UPLOAD_ROOT)
    uploaded = request.files['audio']
    temp_path = os.path.join(temp_dir, secure_filename(uploaded.filename or 'query.wav'))
    uploaded.save(temp_path)
    try:
        return jsonify({'results': semantic_search_audio_by_file(temp_path, top_k=top_k, index_name=index_name, metadata_path=metadata_path)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/user/indexed-files', methods=['GET'])
def list_indexed_files():
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'files': [], 'total_files': 0})
    session_dir, _, metadata_path = _user_paths(session_id)
    if not os.path.exists(session_dir):
        return jsonify({'files': [], 'total_files': 0})

    indexed = get_indexed_files(metadata_path=metadata_path)
    files = []
    for audio_path in indexed:
        files.append({'filename': os.path.basename(audio_path), 'path': audio_path})
    return jsonify({'files': files, 'total_files': len(files)})


@app.route('/api/audio/<session_id>/<path:filename>')
def stream_audio(session_id, filename):
    session_dir, _, _ = _user_paths(session_id)
    file_path = os.path.join(session_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'file not found'}), 404
    
    # Determine MIME type for audio files
    ext = os.path.splitext(file_path)[1].lower()
    audio_mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
    }
    mime_type = audio_mime_types.get(ext, 'audio/mpeg')
    
    return send_from_directory(session_dir, filename, mimetype=mime_type, as_attachment=False)


@app.route('/api/file')
def serve_file():
    file_path = request.args.get('path', '')
    download = request.args.get('download', '0') == '1'
    resolved_path = _safe_existing_path(file_path)
    if not resolved_path:
        return jsonify({'error': 'invalid file path'}), 400
    
    if not os.path.exists(resolved_path):
        return jsonify({'error': 'file not found'}), 404
    
    # Determine MIME type for audio files
    ext = os.path.splitext(resolved_path)[1].lower()
    audio_mime_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4',
        '.aac': 'audio/aac',
    }
    mime_type = audio_mime_types.get(ext, 'audio/mpeg')
    
    return send_file(
        resolved_path,
        mimetype=mime_type,
        as_attachment=download,
        download_name=os.path.basename(resolved_path),
        conditional=True,
    )


@app.route('/api/user/clear', methods=['POST'])
def clear_user():
    data = request.get_json(force=True) or {}
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'ok': False, 'error': 'no session_id'}), 400

    session_dir, index_name, metadata_path = _user_paths(session_id)
    shutil.rmtree(session_dir, ignore_errors=True)
    for path in (f'{index_name}_clap.faiss', metadata_path):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
    return jsonify({'ok': True})

def run_server(host='0.0.0.0', port=5000):
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == '__main__':
    run_server()
