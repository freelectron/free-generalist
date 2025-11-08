import os 

import yt_dlp

from ..utils.clog import get_logger


logger = get_logger(__name__)


def download_audio_mp3(file_path:str, url:str, target_height:int=144):
    """ 
    Primarily configured and used for YouTube. 
    """
    file_path = f'{file_path}'
    ext = 'mp3'
    ydl_opts = {
    'format': 'bestaudio/best',        # Best audio quality
    'outtmpl': file_path,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': ext,         # Or 'aac', 'm4a', etc.
        'preferredquality': '192',     # Bitrate (e.g., 192k)
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        ydl.download([url])

    return file_path+f".{ext}", info


def download_audio(file_name: str, url: str) -> tuple[str, dict[str,str]]:
    """
    Only supports mp3 from YouTube now. 
    """
    file_path_no_extension = os.path.abspath(file_name)
    file_path, meta = download_audio_mp3(file_path_no_extension, url)
    
    return file_path, meta


# TODO: parameterise the function properly 
def download_video_mp4(file_path:str, url:str, target_height:int=730):
    """ Primarily configured and used for YouTube. """
    file_path = f'{file_path}.mp4'
    ydl_opts = {
        'format': f'bestvideo[height<={target_height}][ext=mp4]',
        'outtmpl': file_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        ydl.download([url])

    return file_path, info


def download_video(file_name: str, url: str) -> tuple[str, dict[str,str]]:
    file_path_no_extension = os.path.abspath(file_name)
    file_path, meta = download_video_mp4(file_path_no_extension, url)
 
    return file_path, meta
