from pydantic import BaseModel
from typing import List


class BlogTitlesGeneratorResponse(BaseModel):
    success: bool
    message: str
    titles: List[str]  # List of titles as strings


class BlogTitles(BaseModel):
    titles: List[str]