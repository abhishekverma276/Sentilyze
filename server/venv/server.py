from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
import re
from transformers import pipeline, AutoTokenizer
import emoji

app = Flask(__name__)
CORS(app)

API_KEY = 'API KEY'  # Replace with your actual YouTube API key

# Initialize the tokenizer and sentiment analysis pipeline
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", tokenizer=tokenizer)

def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokenizer.convert_tokens_to_string(tokens)

def map_label(label):
    if label == "LABEL_0":
        return "negative"
    elif label == "LABEL_1":
        return "neutral"
    elif label == "LABEL_2":
        return "positive"
    return label

@app.route('/fetch_comments', methods=['POST'])
def fetch_comments():
    data = request.json
    video_id = data.get('video_id')
    if not video_id:
        return jsonify({'error': 'No video ID provided'}), 400

    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        comments = []
        cleaned_comments = []

        # Paginate through comments
        next_page_token = None
        while True:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=100,
                pageToken=next_page_token
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                cleaned_comment = remove_emojis(comment)
                truncated_comment = truncate_text(cleaned_comment, tokenizer)
                cleaned_comments.append(truncated_comment)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        if not cleaned_comments:
            return jsonify({'error': 'No comments found'}), 400

        # Check the length of cleaned_comments
        print(f"Number of cleaned comments: {len(cleaned_comments)}")

        sentiments = sentiment_pipeline(cleaned_comments)

        # Check the length of sentiments
        print(f"Number of sentiments: {len(sentiments)}")

        # Ensure the number of sentiments matches the number of comments
        if len(cleaned_comments) != len(sentiments):
            return jsonify({'error': 'Mismatch between comments and sentiment analysis'}), 500

        positive_words = []
        negative_words = []

        for comment, sentiment in zip(cleaned_comments, sentiments):
            sentiment_label = map_label(sentiment['label'])
            words = re.findall(r'\w+', comment)
            if sentiment_label == 'positive':
                positive_words.extend(words)
            elif sentiment_label == 'negative':
                negative_words.extend(words)

        positive_word_freq = {word: positive_words.count(word) for word in set(positive_words) if word not in negative_words}
        negative_word_freq = {word: negative_words.count(word) for word in set(negative_words) if word not in positive_words}

        # Sort positive and negative words by frequency
        sorted_positive_words = dict(sorted(positive_word_freq.items(), key=lambda item: item[1], reverse=True))
        sorted_negative_words = dict(sorted(negative_word_freq.items(), key=lambda item: item[1], reverse=True))

        return jsonify({
            'comments': comments,
            'sentiments': [{'label': map_label(s['label']), 'score': s['score']} for s in sentiments],
            'positive_words': sorted_positive_words,
            'negative_words': sorted_negative_words
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
