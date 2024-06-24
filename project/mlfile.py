""" Video Recommender system"""

import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def parse_data(json_file):
    "reads a json file"
    with open(json_file) as f:
        data = json.load(f)
    return data['users'], data['videos']

def calculate_similarity(videos):
    'Calculate the cosine similarity between videos based on their categories and tags'
    count_vectorizer = CountVectorizer().fit_transform(
        [video['category'] + ' ' + ' '.join(video['tags']) for video in videos]
    )
    return cosine_similarity(count_vectorizer)

def recommend_videos(user_id, n, users, videos, video_similarity):
    'recommend videos to a user based on their watch history and video similarities'
    user = next((user for user in users if user['user_id'] == user_id), None)
    if not user:
        return []

    watch_history = set(user['watch_history'])
    scores = [0] * len(videos)
    
    for other_user in users:
        if other_user['user_id'] == user_id:
            continue
        for video_id in other_user['watch_history']:
            if video_id not in watch_history:
                scores[video_id - 101] += 1
                
    for video_id in watch_history:
        for i, score in enumerate(video_similarity[video_id - 101]):
            if videos[i]['video_id'] not in watch_history:
                scores[i] += score

    recommended_video_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    recommended_video_ids = [videos[i]['video_id'] for i in recommended_video_ids[:n]]
    
    return recommended_video_ids

def main(user_id, n):
    'Main function to parse data, calculate video similarities, and print video recommendations'
    users, videos = parse_data("data.json")
    video_similarity = calculate_similarity(videos)
    recommendations = recommend_videos(user_id, n, users, videos, video_similarity)
    print(recommendations)

if __name__ == "__main__":
    user_id = 1
    n = 2  
    main(user_id, n)
    