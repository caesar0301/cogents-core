from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

# ============================================================================
# Message System Definitions
# ============================================================================


def _truncate(text: str, max_length: int = 50) -> str:
    """Truncate text to max_length characters, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _format_image_url(url: str, max_length: int = 50) -> str:
    """Format image URL for display, truncating if necessary."""
    if url.startswith("data:"):
        # Base64 image
        media_type = url.split(";")[0].split(":")[1] if ";" in url else "image"
        return f"<base64 {media_type}>"
    else:
        # Regular URL
        return _truncate(url, max_length)


SupportedImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]


class ContentPartTextParam(BaseModel):
    text: str
    type: Literal["text"] = "text"

    def __str__(self) -> str:
        return f"Text: {_truncate(self.text)}"

    def __repr__(self) -> str:
        return f"ContentPartTextParam(text={_truncate(self.text)})"


class ContentPartRefusalParam(BaseModel):
    refusal: str
    type: Literal["refusal"] = "refusal"

    def __str__(self) -> str:
        return f"Refusal: {_truncate(self.refusal)}"

    def __repr__(self) -> str:
        return f"ContentPartRefusalParam(refusal={_truncate(repr(self.refusal), 50)})"


class ImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] = "auto"
    media_type: SupportedImageMediaType = "image/png"

    def __str__(self) -> str:
        url_display = _format_image_url(self.url)
        return f"Image[{self.media_type}, detail={self.detail}]: {url_display}"

    def __repr__(self) -> str:
        url_repr = _format_image_url(self.url, 30)
        return f"ImageURL(url={repr(url_repr)}, detail={repr(self.detail)}, media_type={repr(self.media_type)})"


class ContentPartImageParam(BaseModel):
    image_url: ImageURL
    type: Literal["image_url"] = "image_url"

    def __str__(self) -> str:
        return str(self.image_url)

    def __repr__(self) -> str:
        return f"ContentPartImageParam(image_url={repr(self.image_url)})"


class Function(BaseModel):
    arguments: str
    name: str

    def __str__(self) -> str:
        args_preview = _truncate(self.arguments, 80)
        return f"{self.name}({args_preview})"

    def __repr__(self) -> str:
        args_repr = _truncate(repr(self.arguments), 50)
        return f"Function(name={repr(self.name)}, arguments={args_repr})"


class ToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"] = "function"

    def __str__(self) -> str:
        return f"ToolCall[{self.id}]: {self.function}"

    def __repr__(self) -> str:
        return f"ToolCall(id={repr(self.id)}, function={repr(self.function)})"


class _MessageBase(BaseModel):
    """Base class for all message types"""

    role: Literal["user", "system", "assistant"]
    cache: bool = False


class UserMessage(_MessageBase):
    role: Literal["user"] = "user"
    content: Union[str, List[Union[ContentPartTextParam, ContentPartImageParam]]]
    name: Optional[str] = None

    @property
    def text(self) -> str:
        """Automatically parse the text inside content"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return "\n".join([part.text for part in self.content if hasattr(part, "text") and part.type == "text"])
        else:
            return ""

    def __str__(self) -> str:
        return f"UserMessage(content={_truncate(self.text)})"

    def __repr__(self) -> str:
        return f"UserMessage(content={repr(_truncate(self.text))})"


class SystemMessage(_MessageBase):
    role: Literal["system"] = "system"
    content: Union[str, List[ContentPartTextParam]]
    name: Optional[str] = None

    @property
    def text(self) -> str:
        """Automatically parse the text inside content"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            return "\n".join([part.text for part in self.content if hasattr(part, "text") and part.type == "text"])
        else:
            return ""

    def __str__(self) -> str:
        return f"SystemMessage(content={_truncate(self.text)})"

    def __repr__(self) -> str:
        return f"SystemMessage(content={repr(_truncate(self.text))})"


class AssistantMessage(_MessageBase):
    role: Literal["assistant"] = "assistant"
    content: Optional[Union[str, List[Union[ContentPartTextParam, ContentPartRefusalParam]]]] = None
    name: Optional[str] = None
    refusal: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)

    @property
    def text(self) -> str:
        """Automatically parse the text inside content"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text = ""
            for part in self.content:
                if hasattr(part, "text") and part.type == "text":
                    text += part.text
                elif hasattr(part, "refusal") and part.type == "refusal":
                    text += f"[Refusal] {part.refusal}"
            return text
        else:
            return ""

    def __str__(self) -> str:
        return f"AssistantMessage(content={_truncate(self.text)})"

    def __repr__(self) -> str:
        return f"AssistantMessage(content={repr(_truncate(self.text))})"


BaseMessage = Union[UserMessage, SystemMessage, AssistantMessage]
