from duckduckgo_search import DDGS, AsyncDDGS
from dotenv import load_dotenv
from urllib.parse import urlparse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import os
import aiohttp
import requests
import json
from ...config_loader import config

class SearchResult(BaseModel):
    URL: HttpUrl
    Domain: Optional[str] = None
    Title: Optional[str] = None
    Description: Optional[str] = None


def get_domain_from_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc



# https://serpapi.com/ 申请 key
VALUE_SERP_API_KEY = config.get("VALUE_SERP_API_KEY")

SERPER_API_KEY = config.get("SERPER_API_KEY")


def search_with_serper_api0(query, num_results=50):
    url = "https://serpapi.com/search"
    payload = json.dumps({"q": query, "engine":"google", "api_key":SERPER_API_KEY, "num": num_results})
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    try:
        results = response.json()
    except requests.exceptions.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
    #results = response.json()

    if response.status_code == 200:
        results = response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")

    print(results)
    print(payload)

    # response = requests.post(url, headers=headers, data=payload)
    # results = response.json()

    organic_results = results.get("organic", [])
    # Extract desired data from the organic results
    search_results = []
    for result in organic_results:
        search_results.append(
            SearchResult(
                URL=result.get("link"),
                Domain=get_domain_from_url(result.get("link")),
                Title=result.get("title"),
                Description=result.get("snippet"),
            )
        )
    return search_results

def search_with_serper_api(query, num_results=50):
    url = "https://serpapi.com/search" 
    # 定义请求的参数
    params = {
        "engine": "baidu",
        "q": query,
        "api_key": SERPER_API_KEY  # 替换成你的实际API密钥
    }

    # 发送GET请求
    response = requests.get(url, params=params)

    # 检查请求是否成功
    if response.status_code == 200:
        try:
            # 解析响应的JSON数据
            results = response.json()
        except json.JSONDecodeError as e:
            # 如果响应不是有效的JSON，打印错误信息
            print(f"Failed to decode JSON: {e}")
    else:
        print(f"Request failed with status code: {response.status_code}")

    

    # response = requests.post(url, headers=headers, data=payload)
    # results = response.json()

    organic_results = results.get("organic_results", [])
    # Extract desired data from the organic results
    search_results = []
    for result in organic_results:
        search_results.append(
            SearchResult(
                URL=result.get("link"),
                Domain=get_domain_from_url(result.get("link")),
                Title=result.get("title"),
                Description=result.get("snippet"),
            )
        )
    return search_results


async def search_with_serper_api_async(
    query: str, num_results: int = 50
) -> List[SearchResult]:
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=payload) as response:
            results = await response.json()

    organic_results = results.get("organic", [])

    # Extract desired data from the organic results
    search_results = []
    for result in organic_results:
        search_results.append(
            SearchResult(
                URL=result.get("link"),
                Domain=get_domain_from_url(result.get("link")),
                Title=result.get("title"),
                Description=result.get("snippet"),
            )
        )

    return search_results


async def search_with_value_serp_async(keyword, num_results=50) -> List[SearchResult]:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                "https://api.valueserp.com/search",
                params={
                    "q": keyword,
                    "num": num_results,
                    "api_key": VALUE_SERP_API_KEY,
                },
            ) as response:
                response.raise_for_status()
                results = await response.json()
                search_results = []
                for result in results.get("organic_results", [])[:num_results]:
                    search_results.append(
                        SearchResult(
                            URL=result.get("link"),
                            Domain=get_domain_from_url(result.get("link")),
                            Title=result.get("title"),
                            Description=result.get("snippet"),
                        )
                    )
                return search_results
        except aiohttp.ClientError as e:
            return []


def search_with_value_serp(keyword, num_results=50) -> List[SearchResult]:
    try:
        response = requests.get(
            "https://api.valueserp.com/search",
            params={"q": keyword, "num": num_results, "api_key": VALUE_SERP_API_KEY},
        )
        response.raise_for_status()
        results = response.json()
        search_results = []
        for result in results.get("organic_results", [])[:num_results]:
            search_results.append(
                SearchResult(
                    URL=result.get("link"),
                    Domain=get_domain_from_url(result.get("link")),
                    Title=result.get("title"),
                    Description=result.get("snippet"),
                )
            )
        return search_results
    except requests.RequestException as e:
        return []


async def search_with_duck_duck_go_async(query, max_results=50) -> List[SearchResult]:
    """
    Perform an asynchronous search using the DuckDuckGo search engine.

    Args:
    query (str): The search query string.
    max_results (int, optional): The maximum number of results to return. Defaults to 50.

    Returns:
    str: A JSON string containing the search results, each result being a dictionary with URL, Title, and Description.
    """
    async with AsyncDDGS() as ddgs:
        results = []
        async for r in ddgs.text(query, max_results=max_results):
            results.append(r)
        result_data = []
        for result in results:
            # Ensure all keys exist to avoid key errors
            url = result.get("href", None)
            title = result.get("title", None)
            description = result.get("body", None)
            if url:
                domain = get_domain_from_url(url)
                result_data.append(
                    SearchResult(
                        URL=url, Domain=domain, Title=title, Description=description
                    )
                )
            else:
                result_data.append(
                    SearchResult(URL=url, Title=title, Description=description)
                )

        return result_data


def search_with_duck_duck_go(query: str, max_results: int = 10) -> List[SearchResult]:
    """
    Perform a synchronous search using the DuckDuckGo search engine.

    Args:
        query (str): The search query string.
        max_results (int, optional): The maximum number of results to return. Defaults to 50.

    Returns:
        List[SearchResult]: A list of SearchResult objects, each containing URL, Title, and Description from the search results.
    """
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=max_results)]
        result_data = []
        for result in results:
            url = result.get("href", None)
            title = result.get("title", None)
            description = result.get("body", None)
            if url:
                domain = get_domain_from_url(url)
                result_data.append(
                    SearchResult(
                        URL=url, Domain=domain, Title=title, Description=description
                    )
                )
            else:
                result_data.append(
                    SearchResult(URL=url, Title=title, Description=description)
                )

        return result_data
