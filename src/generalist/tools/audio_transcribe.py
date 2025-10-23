import whisper

from .. import logging


logger = logging.getLogger(__name__)


def transcribe_mp3_file(filepath: str) -> str:
    """
    Get text transcription (just text and nothing else) from the mp3 file provided. 

    Returns:
        str: speech text that is included in audio.
    """
    transcriber = whisper.load_model("tiny")
    transcription = transcriber.transcribe(filepath)

    return transcription["text"] 