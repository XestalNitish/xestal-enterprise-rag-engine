import os
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Dict, List
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.tools import tool

@tool
def youtube(link: str) -> str:
    """
    Extracts and loads the transcript from a YouTube video URL as LangChain Document objects.   
    Args:
        link (str): The URL of the YouTube video.      
    Returns:
        str: A string containing the video transcript.
    """
    try:
        loader = YoutubeLoader.from_youtube_url(
            link,
            add_video_info=True,
            language=['hi', 'en']
        )
        docs = loader.load()
        print("Video loaded successfully!")
        full_transcript = "\n\n".join([doc.page_content for doc in docs])
        return full_transcript
    except Exception as e:
        error_msg = f"[ERROR] Extraction failed for {link}:{str(e)}"
        print(error_msg)
        return error_msg
