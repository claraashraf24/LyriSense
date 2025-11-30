const API_BASE = 'http://127.0.0.1:8000';

const textInput = document.getElementById('user-text');
const btn = document.getElementById('recommend-btn');
const emotionDiv = document.getElementById('emotion-output');
const songsDiv = document.getElementById('songs-output');

// NEW: strict emotion checkbox (optional)
const strictCheckbox = document.getElementById('strict-emotion');

const artistInput = document.getElementById('artist-filter');
const sortSelect = document.getElementById('sort-by');

async function callRecommend() {
  const text = textInput.value.trim();
  if (!text) {
    alert('Please write how you feel first ✍️');
    return;
  }

  emotionDiv.textContent = 'Thinking...';
  songsDiv.innerHTML = '';

  // NEW: if checkbox exists, use it; otherwise default to false
  const sameEmotionOnly = strictCheckbox ? strictCheckbox.checked : false;

  const artist = artistInput ? artistInput.value.trim() : '';
  const sortBy = sortSelect ? sortSelect.value : 'similarity';

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
      const err = await res.json();
      throw new Error(err.detail || 'Error from API');
    }

    const data = await res.json();

    // Show detected emotion
    const confPct = (data.user_confidence * 100).toFixed(1);
    emotionDiv.textContent = `Detected emotion: ${data.user_emotion} (${confPct}% confident)`;

    // Show recommendations
    if (!data.recommendations || data.recommendations.length === 0) {
      songsDiv.textContent = 'No songs found, sorry.';
      return;
    }

    const list = document.createElement('ul');
    list.className = 'songs-list';

    data.recommendations.forEach((song) => {
      const li = document.createElement('li');
      li.className = 'song-item';

      const simPct = (song.similarity * 100).toFixed(1);
      const emoConfPct = (song.song_emotion_conf * 100).toFixed(1);

      li.innerHTML = `
        <div class="song-title">${song.title}</div>
        <div class="song-artist">${song.artist}${
        song.album
          ? " — <span class='song-album'>" + song.album + '</span>'
          : ''
      }</div>

        <div class="song-meta">
          <span>Song emotion: ${
            song.song_emotion || 'unknown'
          } (${emoConfPct}%)</span>
          <span>Similarity: ${simPct}%</span>
        </div>

        ${
          song.lyric_snippet
            ? `<div class="song-lyrics">"<em>${song.lyric_snippet}</em>"</div>`
            : ''
        }
      `;

      list.appendChild(li);
    });

    songsDiv.appendChild(list);
  } catch (err) {
    console.error(err);
    emotionDiv.textContent = 'Error calling API.';
    songsDiv.textContent = err.message;
  }
}

if (btn) {
  btn.addEventListener('click', callRecommend);
}
