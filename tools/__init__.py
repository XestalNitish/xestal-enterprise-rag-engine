from .links import web_reader, browser_agent
from .pdf import pdf_reader
from .docs import docx_reader
from .video import video_reader
from .youtubeloder import youtube

__all__ = [
    "web_reader",
    "browser_agent",
    "pdf_reader",
    "docx_reader",
    "video_reader",
    "youtube"
]