import os 

import yt_dlp
import whisper

from clog import get_logger


logger = get_logger(__name__)


def transcribe_mp3(file_path: str) -> str:
    """
    This tool get a text transcription from an mp3 file provided.

    Returns:
        str: speech text from the audio.
    """
    transcriber = whisper.load_model("tiny")
    transcription = transcriber.transcribe(file_path)

    return transcription["text"]


def download_audio_mp3(file_path: str, url: str, target_height: int = 144):
    """Download audio from a URL and convert to MP3 format.

    Primarily configured and used for YouTube videos.

    Args:
        file_path (str): Path where the audio file should be saved (without extension).
        url (str): URL of the video to download audio from.
        target_height (int, optional): Target video height parameter. Defaults to 144.
            Note: This parameter is not currently used in the audio download logic.

    Returns:
        tuple[str, dict]: A tuple containing:
            - str: The full file path of the downloaded audio file (with .mp3 extension).
            - dict: Metadata information about the downloaded audio extracted by yt-dlp.
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


def download_audio(file_name: str, url: str) -> tuple[str, dict[str, str]]:
    """Download audio from a URL and save as MP3.

    Args:
        file_name (str): Name/path for the output file (without extension).
        url (str): URL of the video to download audio from.

    Returns:
        tuple[str, dict[str, str]]: A tuple containing:
            - str: The absolute file path of the downloaded audio file.
            - dict: Metadata information about the downloaded audio.
    """
    file_path_no_extension = os.path.abspath(file_name)
    file_path, meta = download_audio_mp3(file_path_no_extension, url)
    
    return file_path, meta


# TODO: parameterise the function properly
def download_video_mp4(file_path: str, url: str, target_height: int = 730):
    """Download video from a URL in MP4 format with specified height.

    Primarily configured and used for YouTube videos.

    Args:
        file_path (str): Path where the video file should be saved (without extension).
        url (str): URL of the video to download.
        target_height (int, optional): Maximum video height in pixels. Defaults to 730.
            The function will download the best quality video that doesn't exceed this height.

    Returns:
        tuple[str, dict]: A tuple containing:
            - str: The full file path of the downloaded video file (with .mp4 extension).
            - dict: Metadata information about the downloaded video extracted by yt-dlp.
    """
    file_path = f'{file_path}.mp4'
    ydl_opts = {
        'format': f'bestvideo[height<={target_height}][ext=mp4]',
        'outtmpl': file_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        ydl.download([url])

    return file_path, info


def download_video(file_name: str, url: str) -> tuple[str, dict[str, str]]:
    """Download video from a URL and save as MP4.

    Args:
        file_name (str): Name/path for the output file (without extension).
        url (str): URL of the video to download.

    Returns:
        tuple[str, dict[str, str]]: A tuple containing:
            - str: The absolute file path of the downloaded video file.
            - dict: Metadata information about the downloaded video.
    """
    file_path_no_extension = os.path.abspath(file_name)
    file_path, meta = download_video_mp4(file_path_no_extension, url)

    return file_path, meta
