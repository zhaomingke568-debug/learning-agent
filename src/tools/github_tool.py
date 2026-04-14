import os
from github import Github
from typing import List

def search_github(query: str, max_results: int = 3) -> List[dict]:
    """Search GitHub for repositories matching the query."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        # Fallback to no-token search if token is missing
        g = Github()
    else:
        g = Github(token)
    
    repositories = g.search_repositories(query=query, sort='stars', order='desc')
    
    results = []
    count = 0
    for repo in repositories:
        if count >= max_results:
            break
        results.append({
            "name": repo.full_name,
            "description": repo.description,
            "url": repo.html_url,
            "stars": repo.stargazers_count,
            "last_update": repo.updated_at.strftime("%Y-%m-%d")
        })
        count += 1
    return results
