import os
from googleapiclient.discovery import build
from typing import List

def search_youtube(query: str, max_results: int = 3) -> List[dict]:
    """Search YouTube for videos matching the query."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return []

    youtube = build("youtube", "v3", developerKey=api_key)
    
    request = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=max_results,
        type="video",
        order="relevance"
    )
    response = request.execute()
    
    results = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        snippet = item["snippet"]
        results.append({
            "id": video_id,
            "title": snippet["title"],
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "description": snippet["description"],
            "channel": snippet["channelTitle"]
        })
    return results
