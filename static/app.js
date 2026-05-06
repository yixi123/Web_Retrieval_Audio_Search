document.addEventListener('DOMContentLoaded', () => {
  const tabs = document.querySelectorAll('.tab');
  const panels = document.querySelectorAll('.tab-pane');

  const searchResults = document.getElementById('searchResults');
  const userResults = document.getElementById('userResults');
  const userFilesList = document.getElementById('userFilesList');
  const uploadProgress = document.getElementById('uploadProgress');
  const sessionIdInput = document.getElementById('sessionId');

  // Cookie management for session persistence
  function getCookie(name) {
    const nameEQ = name + '=';
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
      cookie = cookie.trim();
      if (cookie.indexOf(nameEQ) === 0) {
        return decodeURIComponent(cookie.substring(nameEQ.length));
      }
    }
    return null;
  }

  function setCookie(name, value, days = 365) {
    const date = new Date();
    date.setTime(date.getTime() + days * 24 * 60 * 60 * 1000);
    const expires = 'expires=' + date.toUTCString();
    document.cookie = name + '=' + encodeURIComponent(value) + ';' + expires + ';path=/';
  }

  function generateSessionId() {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  }

  // Initialize session ID from cookie or create new one
  let currentSessionId = getCookie('audioSearchSessionId');
  if (!currentSessionId) {
    currentSessionId = generateSessionId();
    setCookie('audioSearchSessionId', currentSessionId);
  }
  sessionIdInput.value = currentSessionId;

  function setStatus(container, message, isError = false) {
    container.innerHTML = '';
    const div = document.createElement('div');
    div.className = isError ? 'status error' : 'status';
    div.textContent = message;
    container.appendChild(div);
  }

  function displayFileName(result) {
    return result.filename || result.audio_file.split(/[\\/]/).pop();
  }

  function setProgress(container, label, percent) {
    container.innerHTML = '';
    
    // Bootstrap card container for progress
    const card = document.createElement('div');
    card.className = 'card border-info border-2 mb-3';
    
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body p-3';
    
    // Progress label
    const text = document.createElement('div');
    text.className = 'small fw-semibold mb-2 text-info';
    text.textContent = label;

    // Bootstrap progress container
    const progressDiv = document.createElement('div');
    progressDiv.className = 'progress';
    progressDiv.setAttribute('role', 'progressbar');
    progressDiv.setAttribute('aria-valuenow', Math.max(0, Math.min(100, percent)));
    progressDiv.setAttribute('aria-valuemin', '0');
    progressDiv.setAttribute('aria-valuemax', '100');
    progressDiv.style.height = '1.5rem';

    // Progress bar
    const bar = document.createElement('div');
    bar.className = 'progress-bar';
    bar.style.width = Math.max(0, Math.min(100, percent)) + '%';
    bar.textContent = Math.max(0, Math.min(100, percent)) + '%';

    progressDiv.appendChild(bar);
    cardBody.appendChild(text);
    cardBody.appendChild(progressDiv);
    card.appendChild(cardBody);
    container.appendChild(card);
  }

  function formatResultRow(result, kind) {
    const row = document.createElement('div');
    row.className = 'result-row';

    const title = document.createElement('div');
    title.className = 'result-title';
    title.textContent = displayFileName(result);

    const meta = document.createElement('div');
    meta.className = 'result-meta';
    if (typeof result.aggregated_score === 'number') {
      const matchCount = result.num_matched_segments ?? (result.best_matches ? result.best_matches.length : 0);
      meta.textContent = `Aggregated score ${result.aggregated_score} | Matched segments ${matchCount}`;
    } else {
      meta.textContent = `Time ${result.start_time}s - ${result.end_time}s | Similarity ${result.similarity_score}`;
    }

    if (Array.isArray(result.best_matches) && result.best_matches.length > 0) {
      const details = document.createElement('div');
      details.className = 'result-details';
      details.textContent = result.best_matches
        .slice(0, 3)
        .map(match => {
          const [start, end] = match.query_time || [];
          return `Query ${start?.toFixed?.(2) ?? start}s-${end?.toFixed?.(2) ?? end}s -> ${match.indexed_start}s-${match.indexed_end}s (${match.similarity_score})`;
        })
        .join(' | ');
      row.appendChild(title);
      row.appendChild(meta);
      row.appendChild(details);
    } else {
      row.appendChild(title);
      row.appendChild(meta);
    }
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.preload = 'none';
    audio.style.width = '100%';
    
    const audioUrl = kind === 'user'
      ? `/api/audio/${encodeURIComponent(sessionIdInput.value)}/${encodeURIComponent(result.filename || result.audio_file.split(/[\\/]/).pop())}`
      : `/api/file?path=${encodeURIComponent(result.audio_file)}`;
    
    console.log(`Loading ${kind} audio: ${audioUrl}`);
    audio.src = audioUrl;
    
    audio.addEventListener('error', (e) => {
      console.error(`Audio playback error for ${audioUrl}:`, audio.error, e);
      const errorMsg = document.createElement('div');
      errorMsg.className = 'audio-error';
      errorMsg.textContent = `Audio unavailable (${audio.error?.code || 'unknown error'})`;
      row.replaceChild(errorMsg, audio);
    });
    
    audio.addEventListener('loadstart', () => {
      console.log(`Started loading: ${audioUrl}`);
    });

    const download = document.createElement('a');
    download.textContent = 'Download';
    download.className = 'download-link';
    download.href = kind === 'user'
      ? audio.src
      : `/api/file?path=${encodeURIComponent(result.audio_file)}&download=1`;
    download.target = '_blank';
    download.rel = 'noopener noreferrer';

    row.appendChild(audio);
    row.appendChild(download);
    return row;
  }

  function renderResults(container, results, kind) {
    container.innerHTML = '';
    if (!results || results.length === 0) {
      setStatus(container, 'No matches found.');
      return;
    }

    results.forEach(result => {
      container.appendChild(formatResultRow(result, kind));
    });
  }

  async function postJson(url, body) {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }
    return response.json();
  }

  async function postForm(url, formData) {
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }
    return response.json();
  }

  async function indexSessionFiles(sessionId, totalFiles, filesToIndex = []) {
    const response = await fetch('/api/user/index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        session_id: sessionId,
        files_to_index: filesToIndex
      }),
    });

    if (!response.ok || !response.body) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let indexedCount = 0;
    let donePayload = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) {
          continue;
        }
        const event = JSON.parse(line);
        if (event.type === 'progress') {
          indexedCount = event.current || indexedCount;
          const percent = event.total > 0 ? Math.round((event.current / event.total) * 100) : 0;
          setProgress(uploadProgress, `Indexing ${event.current}/${event.total} files...`, percent);
        } else if (event.type === 'done') {
          donePayload = event;
        }
      }
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer.trim());
      if (event.type === 'progress') {
        indexedCount = event.current || indexedCount;
        const percent = event.total > 0 ? Math.round((event.current / event.total) * 100) : 0;
        setProgress(uploadProgress, `Indexing ${event.current}/${event.total} files...`, percent);
      } else if (event.type === 'done') {
        donePayload = event;
      }
    }

    if (!donePayload) {
      donePayload = { indexed_count: indexedCount || totalFiles || 0, files: [] };
    }

    return donePayload;
  }

  function postFormWithProgress(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url);

      xhr.upload.onprogress = event => {
        if (event.lengthComputable && typeof onProgress === 'function') {
          onProgress(event.loaded, event.total);
        }
      };

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch (error) {
            reject(new Error('Invalid JSON response from server.'));
          }
        } else {
          try {
            const errorData = JSON.parse(xhr.responseText || '{}');
            reject(new Error(errorData.error || xhr.responseText || `Request failed: ${xhr.status}`));
          } catch (error) {
            reject(new Error(xhr.responseText || `Request failed: ${xhr.status}`));
          }
        }
      };

      xhr.onerror = () => reject(new Error('Network error while uploading files.'));
      xhr.send(formData);
    });
  }

  function activeSessionId() {
    return (sessionIdInput.value || '').trim();
  }

  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(tab => tab.classList.remove('active'));
      btn.classList.add('active');
      panels.forEach(panel => panel.classList.remove('show', 'active'));
      document.getElementById(btn.dataset.tab).classList.add('show', 'active');
    });
  });

  document.getElementById('textSearchBtn').addEventListener('click', async () => {
    const query = document.getElementById('textQuery').value.trim();
    if (!query) {
      setStatus(searchResults, 'Type a text query first.', true);
      return;
    }

    setStatus(searchResults, 'Searching...');
    try {
      const data = await postJson('/api/search/text', { query, top_k: 10 });
      renderResults(searchResults, data.results || [], 'global');
    } catch (error) {
      setStatus(searchResults, `Search failed: ${error.message}`, true);
    }
  });

  document.getElementById('audioSearchBtn').addEventListener('click', async () => {
    const file = document.getElementById('audioQueryFile').files[0];
    if (!file) {
      setStatus(searchResults, 'Select an audio file first.', true);
      return;
    }

    setStatus(searchResults, 'Searching by audio...');
    try {
      const form = new FormData();
      form.append('audio', file);
      form.append('top_k', '10');
      const response = await fetch('/api/search/audio', { method: 'POST', body: form });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      renderResults(searchResults, data.results || [], 'global');
    } catch (error) {
      setStatus(searchResults, `Audio search failed: ${error.message}`, true);
    }
  });

  document.getElementById('uploadBtn').addEventListener('click', async () => {
    const files = document.getElementById('uploadFiles').files;
    if (!files.length) {
      setStatus(uploadProgress, 'Choose one or more audio files first.', true);
      return;
    }

    const form = new FormData();
    for (const file of files) {
      form.append('files', file);
    }
    if (activeSessionId()) {
      form.append('session_id', activeSessionId());
    }

    try {
      setStatus(uploadProgress, 'Saving files to your session...');
      const data = await postForm('/api/user/upload', form);
      sessionIdInput.value = data.session_id;
      setCookie('audioSearchSessionId', data.session_id);
      const totalFiles = data.files ? data.files.length : 0;
      const savedFilenames = (data.files || []).map(f => f.filename);
      setProgress(uploadProgress, `Indexing 0/${totalFiles} files...`, 0);
      const indexed = await indexSessionFiles(data.session_id, totalFiles, savedFilenames);
      const fileListData = await refreshUserFiles();

      const totalIndexed = indexed.indexed_count ?? totalFiles;
      const sessionTotal = fileListData.total_files ?? indexed.session_total_files ?? totalIndexed;
      const duplicates = data.duplicate_files || [];
      const filesUploaded = files.length;
      
      let statusMsg = `✓ Uploaded ${filesUploaded} file(s). Indexed ${totalIndexed} new file(s). Session total: ${sessionTotal}.`;
      if (duplicates.length > 0) {
        statusMsg += ` Skipped duplicates: ${duplicates.join(', ')}.`;
      }
      setStatus(uploadProgress, statusMsg);
    } catch (error) {
      setStatus(uploadProgress, `Upload failed: ${error.message}`, true);
    }
  });

  document.getElementById('userTextSearchBtn').addEventListener('click', async () => {
    const session_id = activeSessionId();
    const query = document.getElementById('userTextQuery').value.trim();
    if (!session_id) {
      setStatus(userResults, 'Upload files first so a user session exists.', true);
      return;
    }
    if (!query) {
      setStatus(userResults, 'Type a text query first.', true);
      return;
    }

    setStatus(userResults, 'Searching user index...');
    try {
      const data = await postJson('/api/user/search/text', { session_id, query, top_k: 10 });
      const normalized = (data.results || []).map(result => ({
        ...result,
        filename: displayFileName(result),
      }));
      renderResults(userResults, normalized, 'user');
    } catch (error) {
      setStatus(userResults, `Search failed: ${error.message}`, true);
    }
  });

  document.getElementById('userAudioSearchBtn').addEventListener('click', async () => {
    const session_id = activeSessionId();
    const file = document.getElementById('userAudioQueryFile').files[0];
    if (!session_id) {
      setStatus(userResults, 'Upload files first so a user session exists.', true);
      return;
    }
    if (!file) {
      setStatus(userResults, 'Select an audio file first.', true);
      return;
    }

    setStatus(userResults, 'Searching user index by audio...');
    try {
      const form = new FormData();
      form.append('session_id', session_id);
      form.append('audio', file);
      form.append('top_k', '10');
      const response = await fetch('/api/user/search/audio', { method: 'POST', body: form });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      const normalized = (data.results || []).map(result => ({
        ...result,
        filename: displayFileName(result),
      }));
      renderResults(userResults, normalized, 'user');
    } catch (error) {
      setStatus(userResults, `Audio search failed: ${error.message}`, true);
    }
  });

  document.getElementById('showFilesBtn').addEventListener('click', async () => {
    const session_id = activeSessionId();
    if (!session_id) {
      setStatus(userFilesList, 'No user session yet. Upload files first.', true);
      return;
    }
    await refreshUserFiles();
  });

  async function refreshUserFiles() {
    const session_id = activeSessionId();
    userFilesList.innerHTML = '';
    if (!session_id) {
      setStatus(userFilesList, 'No user index yet. Upload files to create one.');
      return { total_files: 0, files: [] };
    }

    try {
      const response = await fetch(`/api/user/indexed-files?session_id=${encodeURIComponent(session_id)}`);
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      const files = data.files || [];
      const totalFiles = data.total_files ?? files.length;
      if (!files.length) {
        setStatus(userFilesList, `No indexed files found yet. Total files: ${totalFiles}`);
        return { total_files: totalFiles, files: [] };
      }

      const summary = document.createElement('div');
      summary.className = 'status info mb-3';
      summary.textContent = `Indexed files (${totalFiles})`;
      userFilesList.appendChild(summary);

      files.forEach(file => {
        const item = document.createElement('div');
        item.className = 'user-file-item';

        const name = document.createElement('strong');
        name.textContent = file.filename;

        const controls = document.createElement('div');
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.src = `/api/audio/${encodeURIComponent(session_id)}/${encodeURIComponent(file.filename)}`;

        const download = document.createElement('a');
        download.className = 'download-link';
        download.textContent = 'Download';
        download.href = audio.src;
        download.target = '_blank';
        download.rel = 'noopener noreferrer';

        controls.appendChild(audio);
        controls.appendChild(download);
        item.appendChild(name);
        item.appendChild(controls);
        userFilesList.appendChild(item);
      });
      return { total_files: totalFiles, files };
    } catch (error) {
      setStatus(userFilesList, `Could not load indexed files: ${error.message}`, true);
      return { total_files: 0, files: [] };
    }
  }

  document.getElementById('clearUserBtn').addEventListener('click', async () => {
    const session_id = activeSessionId();
    if (!session_id) {
      setStatus(userFilesList, 'There is no user session to clear.', true);
      return;
    }

    try {
      const response = await fetch('/api/user/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id }),
      });
      if (!response.ok) throw new Error(await response.text());
      const data = await response.json();
      if (data.ok) {
        // Clear index but maintain session ID for future uploads
        userFilesList.innerHTML = '';
        userResults.innerHTML = '';
        setStatus(userFilesList, 'User index cleared.');
      }
    } catch (error) {
      setStatus(userFilesList, `Clear failed: ${error.message}`, true);
    }
  });

  tabs[0].click();
  refreshUserFiles();
});
