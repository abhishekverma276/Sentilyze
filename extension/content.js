(function() {
  const urlParams = new URLSearchParams(window.location.search);
  const videoId = urlParams.get('v');
  if (videoId) {
    chrome.storage.local.set({ videoId: videoId });
  } else {
    console.error('No video ID found in the URL');
  }
})();
