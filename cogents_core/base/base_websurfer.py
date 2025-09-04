from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class ObserveResult(BaseModel):
    selector: str  # XPath selector for the element (prefixed with "xpath=")
    description: str  # Human-readable description of the element
    backendNodeId: int  # Optional Chrome DevTools Protocol node ID
    method: str  # Suggested Playwright interaction method (e.g., "click", "fill")
    arguments: List[str]  # Method arguments for the suggested interaction


class BaseWebPage(ABC):
    @abstractmethod
    async def navigate(self, url: str, **kwargs) -> None:
        """Navigates to the specified URL."""

    @abstractmethod
    async def act(self, instruction: str, observe_results: Optional[ObserveResult] = None, **kwargs) -> Any:
        """
        Executes an action on the page using natural language.
        This would internally use an AI model to interpret the action
        and translate it into Playwright commands.

        Args:
            instruction: The natural language instruction to execute the action.
            observe_results: The observed results to use for the action. If not provided, the action will be executed
                on the entire page.
            kwargs: Additional keyword arguments.
        Returns:
            The result of the action.
        """

    @abstractmethod
    async def extract(
        self, instruction: str, schema: Union[Dict, BaseModel], selector: Optional[str] = None, **kwargs
    ) -> Union[Dict, BaseModel]:
        """
        Extracts structured data from the page based on a Pydantic-like schema.
        This would leverage an AI model to identify and parse data according to the schema.

        Args:
            instruction: The natural language instruction to extract the data.
            schema: The Pydantic-like schema to extract the data.
            selector: The selector to extract the data.
            kwargs: Additional keyword arguments.
        Returns:
            A dictionary containing the extracted data. If schema is a Pydantic model, it will be returned as is.
            Otherwise, it will be returned as a plain text in `{page_text: string}`.
        """

    @abstractmethod
    async def observe(self, instruction: str, with_actions: bool = True, **kwargs) -> List[ObserveResult]:
        """
        Discovers available actions or elements on the page based on a natural language query.
        This would use an AI model to analyze the page and identify relevant elements.

        Args:
            instruction: The natural language instruction to observe the page.
            with_actions: Whether to include suggested interaction methods
            kwargs: Additional keyword arguments.
        Returns:
            A list of dictionaries containing the observed data.
        """


class BaseWebSurfer(ABC):
    """
    Base class for web surfer agents.
    """

    @abstractmethod
    async def launch(self, headless: bool = True, browser_type: str = "chromium", **kwargs) -> BaseWebPage:
        """
        Launches a new browser instance and returns a BaseWebPage.

        Args:
            headless: Whether to run the browser in headless mode.
            browser_type: The type of browser to use.
            kwargs: Additional keyword arguments.
        Returns:
            A BaseWebPage instance.
        """

    @abstractmethod
    async def close(self):
        """Closes the browser instance.

        Returns:
            None.
        """

    @abstractmethod
    async def agent(self, prompt: str, **kwargs) -> Any:
        """
        Automates an entire workflow autonomously based on a high-level natural language prompt.
        This represents the most agentic capability, orchestrating Act, Extract, Observe internally.

        Args:
            prompt: The natural language prompt to execute the action.
            kwargs: Additional keyword arguments.
        Returns:
            An agent instance capable of executing the workflow.
        """
