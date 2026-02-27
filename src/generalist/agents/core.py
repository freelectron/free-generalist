from typing import Type
from dataclasses import dataclass

from .workflows.workflow_coder import CodeWriterExecutorWorkflow
from .workflows.workflow_web_search import DeepWebSearchWorkflow
from ..models.core import llm
from ..tools.text_processing.text_processing import process_text
from ..tools.data_model import Message
from clog import get_logger
from ..tools.text_processing.utils import read_local_file

logger = get_logger(__name__)


class BaseAgent:
    """Base class for all agent capabilities."""
    name: str  # Defines that all subclasses should have a 'name' class attribute
    capability: str # short description of what the agent is supposed to do

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
    
    def run(self, *args, **kwargs) -> Message:
        """
        Execute the main logic of the agent.
        """
        raise NotImplementedError("Method `run` is not implemented")


class AgentDeepWebSearch(BaseAgent):
    """Capability for performing a deep web search."""
    name = "deep_web_search"
    capability = "searches and downloads web information, but does not process the content and retrieves information"

    def __init__(self, activity: str):
        super().__init__(activity=activity)
        self.queries_per_question: int = 1
        self.links_per_query: int = 1

    def run(self) -> Message:
        agent_workflow = DeepWebSearchWorkflow(
            name=self.name,
            agent_capability=self.capability,
            llm=llm,
            context=[],
            task=self.activity,
        )

        final_state = agent_workflow.run()
        logger.info(f" After running {self.name}, the final state's context is:\n{final_state["context"]}")

        # last output will be a content resource with the downloaded search results (e.g., one or multiple web pages)
        last_resource = final_state["context"][-1]
        return Message(
            provided_by=self.name,
            content=last_resource.content,
            link=last_resource.link,
            metadata=last_resource.metadata,
        )


class AgentUnstructuredDataProcessor(BaseAgent):
    """Capability for processing unstructured text."""
    name = "unstructured_data_processing"
    capability = "analyzes or extracts information from the previously downloaded and available locally resources/text"

    def run(self, resources: list[Message]) -> Message:
        resource_contents = []
        for resource in resources:
            content = resource.content
            if resource.link:
                content = read_local_file(resource.link)
            else:
                logger.warning(f"No link available for resource: {resource}")
            resource_contents.append(content)

        text  = "\n".join(resource_contents)
        answers = process_text(self.activity, text, mode="remote")
        logger.info(f" After running {self.name}, the final state answers are :\n{answers}")

        return Message(
            provided_by=self.name,
            content=f"Query: {self.activity}\nAnswers: {answers}",
        )


class AgentCodeWriterExecutor(BaseAgent):
    """Capability for writing and executing python code"""
    name = "code_writing_execution"
    capability = "can write and execute code for analysing files (e.g., csv, parquet) or performing math operations"

    def run(self, resources:list[Message]) -> Message:
        agent_workflow = CodeWriterExecutorWorkflow(
            name=self.name,
            agent_capability=self.capability,
            llm=llm,
            context=resources,
            task=self.activity,
        )
        final_state = agent_workflow.run()
        logger.info(f" After running the agent workflow :\n{final_state}")

        return Message(
            provided_by=self.name,
            content=f"Query: {self.activity} \n Answers: {str(final_state)}",
        )


@dataclass
class AgentPlan:
    """A structured plan outlining the sequence of capabilities and actions."""
    activity: str
    agent: Type[BaseAgent]
    capability_map = {
        AgentDeepWebSearch.name: AgentDeepWebSearch,
        AgentUnstructuredDataProcessor.name: AgentUnstructuredDataProcessor,
        AgentCodeWriterExecutor.name: AgentCodeWriterExecutor,
    }

    @classmethod
    def from_json(cls, json_data: dict) -> "AgentPlan":
        """
        Convert JSON response into a CapabilityPlan with proper capability objects.
        """
        cap_name = json_data["tool"]
        activity = json_data["activity"]

        if cap_name not in cls.capability_map:
            raise ValueError(f"Unknown capability: {cap_name}")

        cap_class = cls.capability_map[cap_name]

        return AgentPlan(activity=activity, agent=cap_class)
