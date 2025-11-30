// script.js

// Base URL for the FastAPI backend.
// For local development:
//   const API_BASE = 'http://127.0.0.1:8000';
// For same-origin deployment (served behind the same domain/port as frontend),
// you can simply use:
const API_BASE = 'http://127.0.0.1:8000';

const textInput = document.getElementById('user-text');
const btn = document.getElementById('recommend-btn');
const emotionDiv = document.getElementById('emotion-output');
const songsDiv = document.getElementById('songs-output');

// Optional: checkbox for "strict same emotion" mode
const strictCheckbox = document.getElementById('strict-emotion');

const artistInput = document.getElementById('artist-filter');
const sortSelect = document.getElementById('sort-by');

function setLoadingState(isLoading) {
  if (!btn) return;
  btn.disabled = isLoading;
  btn.textContent = isLoading ? 'Thinking…' : 'Get Recommendations';
}

function renderEmotionSummary(data) {
  const confPct = (data.user_confidence * 100).toFixed(1);

  // If backend returns top2_emotions, show both
  if (Array.isArray(data.top2_emotions) && data.top2_emotions.length > 1) {
    const [primary, secondary] = data.top2_emotions;
    const primaryPct = (primary.confidence * 100).toFixed(1);
    const secondaryPct = (secondary.confidence * 100).toFixed(1);

    emotionDiv.innerHTML = `
      <div class="emotion-primary">
        Detected emotion: <strong>${data.user_emotion}</strong>
        (${primaryPct}%)
      </div>
      <div class="emotion-secondary">
        Second strongest: <strong>${secondary.name}</strong>
        (${secondaryPct}%)
      </div>
    `;
  } else {
    // Fallback: only primary emotion
    emotionDiv.textContent = `Detected emotion: ${data.user_emotion} (${confPct}% confident)`;
  }
}

function renderSongs(recommendations) {
  if (!recommendations || recommendations.length === 0) {
    songsDiv.textContent =
      'No songs matched your filters. Try removing the artist filter or changing your mood text.';
    return;
  }

  const list = document.createElement('ul');
  list.className = 'songs-list';

  recommendations.forEach((song) => {
    const li = document.createElement('li');
    li.className = 'song-item';

    const simPct = (song.similarity * 100).toFixed(1);
    const emoConfPct = (song.song_emotion_conf * 100).toFixed(1);

    li.innerHTML = `
      <div class="song-title">${song.title}</div>
      <div class="song-artist">
        ${song.artist}${
      song.album ? " — <span class='song-album'>" + song.album + '</span>' : ''
    }
      </div>

      <div class="song-meta">
        <span>
          Song emotion:
          <strong>${song.song_emotion || 'unknown'}</strong>
          (${emoConfPct}%)
        </span>
        <span>
          Similarity:
          <strong>${simPct}%</strong>
        </span>
      </div>

      ${
        song.lyric_snippet
          ? `<div class="song-lyrics">
               "<em>${song.lyric_snippet}</em>"
             </div>`
          : ''
      }
    `;

    list.appendChild(li);
  });

  songsDiv.appendChild(list);
}

async function callRecommend() {
  const text = textInput.value.trim();
  if (!text) {
    alert('Please write how you feel first ✍️');
    return;
  }

  // Reset UI
  emotionDiv.textContent = 'Thinking...';
  songsDiv.classList.remove('placeholder');
  songsDiv.textContent = 'Loading recommendations...';

  const sameEmotionOnly = strictCheckbox ? strictCheckbox.checked : false;
  const artist = artistInput ? artistInput.value.trim() : '';
  const sortBy = sortSelect ? sortSelect.value : 'similarity';

  setLoadingState(true);

  try {
    const res = await fetch(`${API_BASE}/api/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        top_k: 5,
        same_emotion_only: sameEmotionOnly,
        artist: artist || null,
        sort_by: sortBy,
      }),
    });

    if (!res.ok) {
      // Try to parse backend error message if available
      let errMsg = 'Error from API';
      try {
        const err = await res.json();
        if (err && err.detail) errMsg = err.detail;
      } catch (_) {
        // ignore JSON parse error
      }
      throw new Error(errMsg);
    }

    const data = await res.json();

    // Emotion summary (primary + secondary)
    renderEmotionSummary(data);

    // Recommendations
    songsDiv.innerHTML = '';
    renderSongs(data.recommendations);
  } catch (err) {
    console.error(err);
    emotionDiv.textContent = 'Error calling API.';
    songsDiv.textContent = err.message || 'Something went wrong.';
  } finally {
    setLoadingState(false);
  }
}

// Attach events
if (btn) {
  btn.addEventListener('click', callRecommend);
}

// Optional: allow Ctrl+Enter to submit
if (textInput) {
  textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      callRecommend();
    }
  });
}
