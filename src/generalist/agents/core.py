import os
import re
from typing import Optional, Type, Tuple
from dataclasses import dataclass, field

from browser.search.web import BraveBrowser
from ..tools.summarisers import construct_short_answer
from ..tools.text_processing import process_text
from ..tools.web_search import web_search 
from ..tools.data_model import ContentResource, ShortAnswer
from ..tools.code import write_python_eda, run_code, write_python_task
from ..tools.media import download_audio
from ..tools.media import transcribe_mp3
from ..utils import current_function
from clog import get_logger


logger = get_logger(__name__)


@dataclass
class AgentOutput:
    task: str
    # Based on resources or file attachments, a list of short answers to the task
    answers: Optional[list[ShortAnswer]] = None
    # Produced resources
    resources: Optional[list[ContentResource]] = None


class BaseAgent:
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
    
    def run(self, *args, **kwargs) -> AgentOutput:
        """
        Execute the main logic of the agent.
        """
        raise NotImplementedError("Method `run` is not implemented")


class AgentDeepWebSearch(BaseAgent):
    """Capability for performing a deep web search."""
    name = "deep_web_search"

    def __init__(self, activity: str, search_browser: Optional[BraveBrowser] = None):
        super().__init__(activity=activity)
        self.search_browser = search_browser
        self.queries_per_question: int = 1
        self.links_per_query: int = 1

    def run(self) -> AgentOutput:
        search_results = web_search(
            self.activity,
            brave_search_session=self.search_browser
        )
        resources = []
        for search_result in search_results:
            content = search_result["content"]
            search_result = search_result["search_result"]
            resource = ContentResource(
                provided_by=self.name,
                content=content,
                link=search_result.link,
                metadata=search_result.metadata,
            )
            resources.append(resource)

        links = [r.link for r in resources]
        context = f"Downloaded content for: {links}, see resources that are passed to the next task. You should proceed to text processing!"
        answer = [ShortAnswer(answered=True, answer=str(links), clarification=context)]

        return AgentOutput(
            task=self.activity,
            answers=answer,
            resources=resources,
        )


class AgentAudioProcessor(BaseAgent):
    """Capability for processing audio files."""
    name = "audio_processing"

    def run(self, resources: list[ContentResource]):
        
        # FIXME: find a way to transcribe the correct audio file from the resources
        resource = resources[0]

        # downloads the file from the web to local (do not provide extension)
        temp_folder = os.environ.get("TEMP_FILES_FOLDER", "./")
        cleaned_link = re.sub(r'[^A-Za-z0-9]', '', resource.link[-20])
        file_path_no_extension = os.path.join(temp_folder, "audio_processor", cleaned_link)
        file_path, meta = download_audio(file_path_no_extension, resource.link)
        logger.info(f"- {current_function()} -- Downloaded file to {file_path} with meta info {meta}.")

        # transcribes the file and returns just the text
        result = transcribe_mp3(file_path)

        short_answers = [construct_short_answer(self.activity, result)]

        return AgentOutput(
            task=self.activity,
            answers=short_answers
        ) 


class AgentImageProcessor(BaseAgent):
    """Capability for processing image files."""
    name = "image_processing"


class AgentUnstructuredDataProcessor(BaseAgent):
    """Capability for processing unstructured text."""
    name = "unstructured_data_processing"

    def run(self, resources: list[ContentResource]) -> AgentOutput:
        """
        Args:
        ask (str): what data we are analysing 
        """
        resource_contents = [resource.content for resource in resources]

        # Join contents all together to give to a chunk splitter  
        text  = "\n".join(resource_contents)
        answers = process_text(self.activity, text)

        # Remove processed resources
        resources.clear()

        short_answers = [construct_short_answer(self.activity, answer) for answer in answers] 

        return AgentOutput(
            task=self.activity,
            answers=short_answers,
            resources=[]
        )


class AgentCodeWriterExecutor(BaseAgent):
    """Capability for writing and executing python code"""
    name = "code_writing_execution"

    def run(self, resources:list[ContentResource]) -> AgentOutput:
        # Analyse the given resources, determine what the files contain (EDA)
        eda_code = do_eda_table(resources=resources)
        logger.info(f"- {AgentCodeWriterExecutor.name} -- EDA code to be executed:\n{eda_code}")
        eda_result = run_code(eda_code)
        logger.info(f"- {AgentCodeWriterExecutor.name} -- EDA code result:\n{eda_result}")

        task_code = write_python_task(task=self.activity, eda_results=eda_result, resources=resources)
        logger.info(f"- {AgentCodeWriterExecutor.name} -- Final code to be executed:\n{task_code}")
        result = run_code(task_code)
        logger.info(f"- {AgentCodeWriterExecutor.name} -- Final code result:\n{result}")

        short_answers = [construct_short_answer(self.activity, result)]

        return AgentOutput(
            task=self.activity,
            answers=short_answers,
        )


@dataclass
class AgentPlan:
    """A structured plan outlining the sequence of capabilities and actions."""
    subplan: list[Tuple[str, BaseAgent]]
    capability_map = {
        AgentDeepWebSearch.name: AgentDeepWebSearch,
        AgentAudioProcessor.name: AgentAudioProcessor,
        AgentImageProcessor.name: AgentImageProcessor,
        AgentUnstructuredDataProcessor.name: AgentUnstructuredDataProcessor,
        AgentCodeWriterExecutor.name: AgentCodeWriterExecutor,
    }

    @classmethod
    def from_json(cls, json_data: dict) -> "AgentPlan":
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

        return AgentPlan(subplan=subplan)
