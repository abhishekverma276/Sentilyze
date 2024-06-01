document.addEventListener('DOMContentLoaded', function () {
  document.getElementById('fetch-comments').addEventListener('click', async () => {
    const videoUrl = document.getElementById('video-url').value;
    const videoId = getVideoIdFromUrl(videoUrl);

    if (videoId) {
      try {
        const res = await fetch('http://127.0.0.1:5000/fetch_comments', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ video_id: videoId })
        });
        const data = await res.json();
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '';

        if (data.error) {
          resultsDiv.textContent = `Error fetching comments: ${data.error}`;
        } else {
          data.comments.forEach((comment, index) => {
            const sentiment = data.sentiments[index];
            const div = document.createElement('div');
            div.className = 'comment';
            div.innerHTML = `<strong>Comment:</strong> ${comment}<br><strong>Sentiment:</strong> ${sentiment.label}, <strong>Score:</strong> ${sentiment.score.toFixed(2)}`;
            resultsDiv.appendChild(div);
          });

          const positiveWordsDiv = document.createElement('div');
          positiveWordsDiv.innerHTML = `<h3>Most used positive words:</h3><p>${JSON.stringify(data.positive_words, null, 2)}</p>`;
          resultsDiv.appendChild(positiveWordsDiv);

          const negativeWordsDiv = document.createElement('div');
          negativeWordsDiv.innerHTML = `<h3>Most used negative words:</h3><p>${JSON.stringify(data.negative_words, null, 2)}</p>`;
          resultsDiv.appendChild(negativeWordsDiv);
        }
      } catch (error) {
        console.error('Error:', error);
      }
    } else {
      alert('Invalid YouTube video URL');
    }
  });

  function getVideoIdFromUrl(url) {
    const urlParams = new URLSearchParams(new URL(url).search);
    return urlParams.get('v');
  }
});
