import os
import re
from typing import Optional, Type, Tuple
from dataclasses import dataclass, field

from browser.search.web import BraveBrowser
from ..tools.summarisers import construct_short_answer
from ..tools.text_processing import text_process_llm
from ..tools.web_search import web_search 
from ..tools.data_model import ContentResource, ShortAnswer
from ..tools.code import write_python_eda, run_code, write_python_code_task
from ..tools.media_download import download_audio
from ..tools.audio_transcribe import transcribe_mp3_file
from ..utils import current_function
from clog import get_logger


logger = get_logger(__name__)


@dataclass
class AgentCapabilityOutput: 
    # Based on resources or file attachments, a list of short answers to the task 
    answers: Optional[list[ShortAnswer]] = None
    # Produced resources
    resources: Optional[list[ContentResource]] = None


class BaseAgentCapability:
    """Base class for all agent capabilities."""
    name: str  # Defines that all subclasses should have a 'name' class attribute

    def __init__(self, activity: str):
        """
        FIXME: CHANGE AGENT TO BE ACTION AGNOSTIC
               Probably agents just need tools and/or resources as input

        Initializes the capability with a specific activity.

        Args:
            activity (str): The specific action/task to be performed.
        """
        self.activity = activity

    def __repr__(self) -> str:
        """Provides a clean string representation of the object."""
        # 'self.name' correctly accesses the class attribute from the instance
        return f"{self.__class__.__name__}(name='{self.name}', activity='{self.activity}')"
    
    def run(self, *args, **kwargs) -> AgentCapabilityOutput:
        """
        Execute the main logic of the agent.
        """
        raise NotImplementedError("Method `run` is not implemented")


class AgentCapabilityDeepWebSearch(BaseAgentCapability):
    """Capability for performing a deep web search."""
    name = "deep_web_search"

    def __init__(self, activity: str, search_browser: Optional[BraveBrowser] = None):
        super().__init__(activity=activity)
        self.search_browser = search_browser

    def run(self) -> AgentCapabilityOutput:
        return AgentCapabilityOutput(
            answers=None,
            resources=web_search(
                self.activity,
                brave_search_session=self.search_browser
            )
        )


class AgentCapabilityAudioProcessor(BaseAgentCapability):
    """Capability for processing audio files."""
    name = "audio_processing"

    def run(self, resources: list[ContentResource]):
        
        # FIXME: for testing, properly determine how to identify which resource to download 
        resource = None
        for r in resources:
            if r.link.startswith("http"):
                resource = r
                break 

        # downloads the file from the web to local (do not provide extension)
        temp_folder = os.environ.get("TEMP_FILES_FOLDER", "./")
        cleaned_link = re.sub(r'[^A-Za-z0-9]', '', resource.link[-20])
        file_path_no_extension = os.path.join(temp_folder, "audio_processor", cleaned_link)
        file_path, meta = download_audio(file_path_no_extension, resource.link)
        logger.info(f"- {current_function()} -- Downloaded file to {file_path} with meta info {meta}.")

        # transcribes the file and returns just the text
        result = transcribe_mp3_file(file_path)

        short_answers = [construct_short_answer(self.activity, result)] 
        return AgentCapabilityOutput(
            answers=short_answers
        ) 


class AgentCapabilityImageProcessor(BaseAgentCapability):
    """Capability for processing image files."""
    name = "image_processing"


class AgentCapabilityUnstructuredDataProcessor(BaseAgentCapability):
    """Capability for processing unstructured text."""
    name = "unstructured_data_processing"

    def run(self, resources: list[ContentResource]) -> AgentCapabilityOutput:
        """
        Args:
        ask (str): what data we are analysing 
        """
        # FIXME: Handle all resources, not just the ones from web search
        resource_contents = [web_resource.content for web_resource in resources]

        # TODO: make this more robust, also now can only handle resources that have text content
        text  = "; \n\n |  ".join(resource_contents)

        answers = text_process_llm(self.activity, text)
        short_answers = [construct_short_answer(self.activity, answer) for answer in answers] 

        return AgentCapabilityOutput(
            answers=short_answers
        )


class AgentCapabilityCodeWritterExecutor(BaseAgentCapability):
    """Capability for writing and executing python code"""
    name = "code_writing_execution"

    def run(self, resources:list[ContentResource]) -> AgentCapabilityOutput:
        """ 
        TODO: determine how to pass resources in the output here?
        """
        # Analyse the given resources, determine what the files contain (EDA)
        eda_code = write_python_eda(resources)
        eda_result = run_code(eda_code)  
        logger.info(f"- AgentCapabilityCodeWritterExecutor.run -- EDA code to be executed:\n{eda_code}")
        # Given the activity=task and what EDA results determine the final code that would produce the result
        task_code = write_python_code_task(task=self.activity, eda_results=eda_result, resources=resources)
        logger.info(f"- AgentCapabilityCodeWritterExecutor.run -- Final code to be executed:\n{task_code}")
        result = run_code(task_code)
        short_answers = [construct_short_answer(self.activity, result)] 
        
        # TODO: determine whether we want to create new resources here? always store info from running the scripts..
        return AgentCapabilityOutput(
            answers=short_answers
        ) 

@dataclass
class CapabilityPlan:
    """A structured plan outlining the sequence of capabilities and actions."""
    subplan: list[Tuple[str, BaseAgentCapability]]
    capability_map = {
        AgentCapabilityDeepWebSearch.name: AgentCapabilityDeepWebSearch,
        AgentCapabilityAudioProcessor.name: AgentCapabilityAudioProcessor,
        AgentCapabilityImageProcessor.name: AgentCapabilityImageProcessor,
        AgentCapabilityUnstructuredDataProcessor.name: AgentCapabilityUnstructuredDataProcessor,
        AgentCapabilityCodeWritterExecutor.name: AgentCapabilityCodeWritterExecutor,
    }

    @classmethod
    def from_json(cls, json_data: dict) -> "CapabilityPlan":
        """
        Convert JSON response into a CapabilityPlan with proper capability objects.
        """
        subplan = []
        for step in json_data["subplan"]:
            cap_name = step["capability"]
            activity = step["activity"]

            if cap_name not in cls.capability_map:
                raise ValueError(f"Unknown capability: {cap_name}")

            cap_class = cls.capability_map[cap_name]
            subplan.append((activity, cap_class))

        return CapabilityPlan(subplan=subplan)




