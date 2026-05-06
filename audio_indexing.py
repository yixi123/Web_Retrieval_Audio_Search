import os
import json
import numpy as np
import torch
import librosa
import faiss
from typing import List, Dict
import laion_clap

# Configuration
SAMPLE_RATE = 48000
SEGMENT_DURATION = 1  # 1-second segments
EMBEDDING_DIM = 512  # CLAP embedding dimension
SIMILARITY_THRESHOLD = 0  # Minimum similarity for results
JSON_METADATA_PATH = "audio_metadata.json"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
 
# Path to your manually downloaded file
LOCAL_PATH = 'music_speech_audioset.pt'
CLAP_MODEL = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', tmodel='roberta')

# Instead of auto-downloading, provide the local path
try:
    # Try strict loading first
    CLAP_MODEL.load_ckpt(ckpt=LOCAL_PATH)
except RuntimeError:
    # If strict loading fails, load with relaxed constraints
    try:
        # Allow unsafe globals for numpy scalars (needed for older checkpoints)
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except:
        pass
    
    checkpoint = torch.load(LOCAL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        CLAP_MODEL.model.load_state_dict(checkpoint['model'], strict=False)
    else:
        CLAP_MODEL.model.load_state_dict(checkpoint, strict=False)
    print("✅ Loaded checkpoint with non-strict mode (architecture mismatch handled)")

CLAP_MODEL.eval()
CLAP_MODEL.to(DEVICE)

# ----------------------
# Utility Functions
# ----------------------

def init_json_store(metadata_path: str = JSON_METADATA_PATH):
    """Initialize the JSON metadata store if it does not already exist.

    Returns:
        None
    """
    if not os.path.exists(metadata_path):
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump([], f)
 
def save_segments_to_json(segments: List[Dict], metadata_path: str = JSON_METADATA_PATH):
    """Append segment metadata to the JSON metadata store.

    Args:
        segments: A list of dictionaries describing indexed audio segments.

    Returns:
        None
    """
    init_json_store(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing.extend(segments)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
 
def load_audio(audio_path: str) -> np.ndarray:
    """Load an audio file and resample it to mono at the configured sample rate.

    Args:
        audio_path: Path to the audio file to load.

    Returns:
        A one-dimensional NumPy array containing the resampled waveform.
    """
    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    return waveform
 
def split_audio_into_segments(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split audio into 1-second segments with 50% overlap and filter silence.

    Args:
        audio: A mono waveform array sampled at SAMPLE_RATE.

    Returns:
        A tuple of (segments, timestamps) where segments is an array of valid
        audio segments and timestamps contains the corresponding start and end
        times in seconds.
    """
    segment_samples = int(SEGMENT_DURATION * SAMPLE_RATE)
    segments = []
    timestamps = []

    # Handle short audio files (less than 1 second)
    if len(audio) < segment_samples:
        # Calculate RMS on original audio before padding to accurately check for silence
        rms = librosa.feature.rms(y=audio)[0].mean()
        if rms >= 0.005: 
            # Pad the end with zeros to meet the 1-second requirement
            padded_audio = np.pad(audio, (0, segment_samples - len(audio)), mode='constant')
            segments.append(padded_audio)
            # Record the actual length of the sound for the metadata timestamp
            timestamps.append((0.0, len(audio) / SAMPLE_RATE))
        return np.array(segments), np.array(timestamps)
    
    for i in range(0, len(audio) - segment_samples, segment_samples // 2):
        segment = audio[i:i+segment_samples]
        if len(segment) == segment_samples:
            # Skip silent segments
            rms = librosa.feature.rms(y=segment)[0].mean()
            if rms >= 0.005:
                segments.append(segment)
                timestamps.append((i / SAMPLE_RATE, (i + segment_samples) / SAMPLE_RATE))
    
    return np.array(segments), np.array(timestamps)

# ----------------------
# Embedding Functions
# ----------------------
def get_audio_embedding(audio_segment: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Compute a CLAP embedding for an audio segment.

    Args:
        audio_segment: The waveform for a single audio segment.
        sample_rate: Sample rate of the input segment.

    Returns:
        A 512-dimensional NumPy embedding vector.
    """
    # Resample to 48kHz (required by CLAP)
    if sample_rate != 48000:
        import librosa
        audio_segment = librosa.resample(audio_segment, orig_sr=sample_rate, target_sr=48000)

    audio_segment = np.asarray(audio_segment, dtype=np.float32).squeeze()
    if audio_segment.ndim == 0:
        audio_segment = np.expand_dims(audio_segment, axis=0)
    audio_segment = np.expand_dims(audio_segment, axis=0)
    
    with torch.no_grad():
        embedding = CLAP_MODEL.get_audio_embedding_from_data(
            x=audio_segment,
            use_tensor=False
        )
    if hasattr(embedding, 'cpu'):
        embedding = embedding.cpu().numpy()
    return np.asarray(embedding).squeeze()

def get_text_embedding(text_query: str) -> np.ndarray:
    """Compute a CLAP embedding for a text query.

    Args:
        text_query: Natural-language query string.

    Returns:
        A 512-dimensional NumPy embedding vector.
    """
    with torch.no_grad():
        embedding = CLAP_MODEL.get_text_embedding([text_query])
    if hasattr(embedding, 'cpu'):
        embedding = embedding.cpu().numpy()
    return np.asarray(embedding).squeeze()


# ----------------------
# Indexing Functions
# ----------------------
def index_audio_file(
    audio_path: str,
    index_name: str = "audio_segments",
    metadata_path: str = JSON_METADATA_PATH,
    verbose: bool = True,
) -> faiss.Index:
    """Index an audio file for later semantic search.

    The file is loaded, segmented, embedded, added to the FAISS index, and its
    segment metadata is appended to the JSON metadata store.

    Args:
        audio_path: Path to the audio file to index.
        index_name: Base name of the FAISS index files.
        verbose: Whether to print progress and error messages.

    Returns:
        The updated FAISS index, or None if indexing fails or the file is
        already indexed.
    """
    audio_path = os.path.normpath(audio_path) 
    try:
        # Load and split audio
        if not os.path.exists(audio_path):
            if verbose:
                print(f"❌ Error: Audio file not found: {audio_path}")
            return None
        
        # Check if audio file already indexed
        init_json_store(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)

        if audio_path in set(metadata["audio_path"] for metadata in all_metadata):
            if verbose:
                print(f"✅ Audio file {audio_path} is already indexed.")
            return None

        audio = load_audio(audio_path)
        segments, timestamps = split_audio_into_segments(audio)
        if verbose:
            print(f"Split {audio_path} into {len(segments)} valid 1-second segments")
        
        if len(segments) == 0:
            if verbose:
                print(f"⚠️ No valid non-silent segments found in {audio_path}")
            return None
        
        # Extract embeddings
        embeddings = []
        segment_metadata = []
        for i, segment in enumerate(segments):
            emb = get_audio_embedding(segment)
            embeddings.append(emb)
            segment_metadata.append({
                "audio_path": audio_path,
                "start_time": float(timestamps[i][0]),
                "end_time": float(timestamps[i][1]),
            })
        
        # Build FAISS index
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)  # Normalize for cosine similarity
        index_path = f"{index_name}_clap.faiss"
        if os.path.exists(index_path):
            faiss_index = faiss.read_index(index_path)
        else:
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        faiss_index.add(embeddings_np)
        faiss.write_index(faiss_index, f"{index_name}_clap.faiss")
        if verbose:
            print(f"✅ FAISS index saved to {index_name}_clap.faiss")
        
        # Save metadata
        save_segments_to_json(segment_metadata, metadata_path=metadata_path)
        return faiss_index
    
    except Exception as e:
        if verbose:
            print(f"❌ Error processing audio file: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    

# ----------------------
# Search Pipeline
# ----------------------
def semantic_search_audio( 
    query_text: str,
    top_k: int = 10,
    index_name: str = "audio_segments",
    metadata_path: str = JSON_METADATA_PATH,
    merge_adjacent: bool = True, 
    similarity_threshold: float = SIMILARITY_THRESHOLD
 ) -> List[Dict]:
    """Search indexed audio using a natural-language text query.

    Args:
        query_text: Natural-language description of the sound to find.
        top_k: Maximum number of results to return.
        index_name: Base name of the FAISS index files.
        merge_adjacent: Whether to merge adjacent matching segments from the
            same audio file.
        similarity_threshold: Minimum similarity score required for a match.

    Returns:
        A list of matching segment dictionaries with time ranges and similarity
        scores, ranked by score.
    """
    try:
        # Load index and metadata
        index_path = f"{index_name}_clap.faiss"
        if not os.path.exists(index_path):
            print(f"❌ Error: Index {index_path} not found. Index audio files first.")
            return []
        
        faiss_index = faiss.read_index(index_path)
        init_json_store(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
        
        if len(all_metadata) == 0:
            print("❌ Error: No metadata found. Index audio files first.")
            return []
        
        # Get query embedding
        query_emb = get_text_embedding(query_text)
        query_emb = np.expand_dims(query_emb, axis=0)

        faiss.normalize_L2(query_emb)
        # Search FAISS
        candidate_count = faiss_index.ntotal
        similarities, indices = faiss_index.search(query_emb, candidate_count)

        # Format results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if sim < similarity_threshold or idx >= len(all_metadata):
                continue
            meta = all_metadata[idx]
            results.append({
                "audio_file": meta["audio_path"],
                "start_time": round(meta["start_time"], 2),
                "end_time": round(meta["end_time"], 2),
                "similarity_score": round(float(sim), 3)
            })
        results.sort(key=lambda x: (x['audio_file'], x['start_time']))
        # Merge adjacent segments
        if merge_adjacent and results:
            merged = []
            current = results[0]
            for res in results[1:]:
                if (res['audio_file'] == current['audio_file'] and 
                    res['start_time'] <= current['end_time'] + 0.1):
                    current['end_time'] = res['end_time']
                    current['similarity_score'] = max(current['similarity_score'], res['similarity_score'])
                else:
                    merged.append(current)
                    current = res
            merged.append(current)
            results = merged
        
        # Sort results by similarity score descending
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    except Exception as e:
        print(f"❌ Search error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def semantic_search_audio_by_file(
    query_audio_path: str,
    top_k: int = 10,
    index_name: str = "audio_segments",
    metadata_path: str = JSON_METADATA_PATH,
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> List[Dict]:
    """Search indexed audio using an audio query file.

    The query audio is segmented, and for each query segment the best matching
    indexed segment is found per indexed file. Those per-segment maxima are then
    averaged into a final score per file.

    Args:
        query_audio_path: Path to the query audio file
        top_k: Maximum number of ranked files to return.
        index_name: Base name of the FAISS index files.
        metadata_path: Path to the JSON metadata file associated with the index.
        similarity_threshold: Minimum similarity score required for a match.
    
    Returns:
        A list of dictionaries ranked by aggregated score. Each dictionary
        includes the indexed file path, aggregated score, matched segment-level
        details, and per-segment timing information.
    """
    try:
        # Load index and metadata
        index_path = f"{index_name}_clap.faiss"
        if not os.path.exists(index_path):
            print(f"❌ Error: Index {index_path} not found. Index audio files first.")
            return []
        
        if not os.path.exists(query_audio_path):
            print(f"❌ Error: Query audio file not found: {query_audio_path}")
            return []
        
        faiss_index = faiss.read_index(index_path)
        init_json_store(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
        
        if len(all_metadata) == 0:
            print("❌ Error: No metadata found. Index audio files first.")
            return []
        
        # Load and segment query audio
        query_audio = load_audio(query_audio_path)
        query_segments, query_timestamps = split_audio_into_segments(query_audio)
        
        if len(query_segments) == 0:
            print(f"⚠️ No valid segments found in query audio: {query_audio_path}")
            return []
        
        print(f"Query audio segmented into {len(query_segments)} segments")
        
        # Get unique indexed files
        indexed_files = list(set(meta["audio_path"] for meta in all_metadata))
        file_scores = {file: [] for file in indexed_files}
        file_best_matches = {file: [] for file in indexed_files}
        
        # For each query segment, find best match per indexed file
        for seg_idx, query_segment in enumerate(query_segments):
            query_emb = get_audio_embedding(query_segment)
            query_emb = np.expand_dims(query_emb, axis=0)
            faiss.normalize_L2(query_emb)
            
            # Search FAISS for all candidates
            candidate_count = min(len(all_metadata), faiss_index.ntotal)
            similarities, indices = faiss_index.search(query_emb, candidate_count)
            
            # Group results by indexed file and find best match per file
            file_best = {file: (0, None) for file in indexed_files}  # (score, metadata)
            
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= len(all_metadata):
                    continue
                meta = all_metadata[idx]
                audio_file = meta["audio_path"]
                
                if sim > file_best[audio_file][0]:
                    file_best[audio_file] = (float(sim), meta)
            
            # Store results for this query segment
            for audio_file, (score, meta) in file_best.items():
                if score >= similarity_threshold and meta is not None:
                    file_scores[audio_file].append(score)
                    query_start = float(query_timestamps[seg_idx][0])
                    query_end = float(query_timestamps[seg_idx][1])
                    file_best_matches[audio_file].append({
                        "query_segment_idx": seg_idx,
                        "query_time": [query_start, query_end],
                        "similarity_score": round(score, 3),
                        "indexed_start": round(meta["start_time"], 2),
                        "indexed_end": round(meta["end_time"], 2)
                    })
        
        # Calculate aggregated scores (average of max scores per segment)
        results = []
        for audio_file in indexed_files:
            if file_scores[audio_file]:
                avg_score = float(np.mean(file_scores[audio_file]))
                results.append({
                    "audio_file": audio_file,
                    "aggregated_score": round(avg_score, 3),
                    "num_matched_segments": len(file_scores[audio_file]),
                    "segment_scores": [round(s, 3) for s in file_scores[audio_file]],
                    "best_matches": file_best_matches[audio_file]
                })
        
        # Sort by aggregated score descending
        results.sort(key=lambda x: x['aggregated_score'], reverse=True)
        
        return results[:top_k]
    
    except Exception as e:
        print(f"❌ Audio search error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# ----------------------
# Helper Functions
# ----------------------
def list_indexed_files(metadata_path: str = JSON_METADATA_PATH) -> List[str]:
    """Return the set of unique audio files currently present in the index.

    Returns:
        A list of unique indexed audio file paths.
    """
    init_json_store(metadata_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        all_metadata = json.load(f)
    return list(set(meta["audio_path"] for meta in all_metadata))
 
def clear_index(index_name: str = "audio_segments", metadata_path: str = JSON_METADATA_PATH):
    """Remove the FAISS index file and JSON metadata store.

    Args:
        index_name: Base name of the FAISS index files.

    Returns:
        None
    """
    index_path = f"{index_name}_clap.faiss"
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    print("✅ Index cleared successfully")

def remove_audio_file(
    audio_path: str,
    index_name: str = "audio_segments",
    metadata_path: str = JSON_METADATA_PATH,
    verbose: bool = True,
) -> bool:
    """Remove all indexed segments for a single audio file.

    Args:
        audio_path: Path of the indexed audio file to remove.
        index_name: Base name of the FAISS index files.
        verbose: Whether to print progress and error messages.

    Returns:
        True if the file was removed successfully, otherwise False.
    """
    try:
        index_path = f"{index_name}_clap.faiss"
        
        # Check if index exists
        if not os.path.exists(index_path):
            if verbose:
                print(f"❌ Error: Index {index_path} not found")
            return False
        
        # Load metadata
        init_json_store(metadata_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            all_metadata = json.load(f)
        
        # Find indices to remove
        indices_to_remove = [i for i, meta in enumerate(all_metadata) if meta["audio_path"] == audio_path]
        
        if not indices_to_remove:
            if verbose:
                print(f"⚠️ Audio file {audio_path} not found in index")
            return False
        
        # Load current index
        faiss_index = faiss.read_index(index_path)
        
        # Create ID array and remove indices
        ids_to_remove = np.array(indices_to_remove, dtype=np.int64)
        
        # Use remove_ids if available
        faiss_index.remove_ids(ids_to_remove)

        
        # Save updated index
        faiss.write_index(faiss_index, index_path)
        
        # Update metadata by removing entries
        all_metadata = [meta for meta in all_metadata if meta["audio_path"] != audio_path]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print(f"✅ Successfully removed {len(indices_to_remove)} segments from {audio_path}")
        return True
    
    except Exception as e:
        if verbose:
            print(f"❌ Error removing audio file: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
