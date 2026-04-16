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
            "channel": snippet["channelTitle"],
            "transcript": get_youtube_transcript(f"https://www.youtube.com/watch?v={video_id}") or "获取字幕失败"
        })
    return results

from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse

def get_youtube_transcript(video_url: str) -> str:
    """提取视频字幕并合并为长文本"""
    try:
        # 1. 从 URL 中解析出 video_id (例如: dQw4w9WgXcQ)
        parsed_url = urllib.parse.urlparse(video_url)
        video_id = urllib.parse.parse_qs(parsed_url.query).get('v')
        if not video_id:
            return "无法解析视频 ID"
            
        # 2. 调用 API 获取中/英文字幕
        # 优先拉取中文，如果没有则拉取英文并自动翻译
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id[0], languages=['zh-CN', 'en'])
        except Exception as e:
            return f"获取字幕失败: {e}"
        
        # 3. 将字幕字典合并成一段纯文本
        full_text = " ".join([t['text'] for t in transcript_list])
        
        # 4. 视频通常很长，为了防止撑爆 Token，截取前 5000 个字符（或调用小模型压缩）
        return full_text[:5000] 
        
    except Exception as e:
        return f"获取字幕失败: {e}"

