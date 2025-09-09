"""
Independent Prompts Module

A comprehensive and extensible prompt management system for handling various
prompt templates, dynamic variable injection, and multi-format support.

This module is self-contained and doesn't depend on other cogents_core modules.
"""

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

# Try to import external dependencies
try:
    from jinja2 import BaseLoader, Environment, FileSystemLoader

    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False

# Import message types from message.py
from .message import AssistantMessage, BaseMessage, SystemMessage, UserMessage

# ============================================================================
# Prompt Management System
# ============================================================================


class TemplateEngine(Enum):
    """Supported template engines"""

    STRING = "string"  # Python string.Template
    JINJA2 = "jinja2"  # Jinja2 templates
    FORMAT = "format"  # Python str.format()


class PromptMetadata(BaseModel):
    """Metadata for a prompt template"""

    name: str
    description: str = ""
    version: str = "1.0"
    author: str = ""
    created_at: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    required_variables: List[str] = Field(default_factory=list)
    optional_variables: Dict[str, Any] = Field(default_factory=dict)
    template_engine: TemplateEngine = TemplateEngine.JINJA2


class MessageTemplate(BaseModel):
    """Template for a single message"""

    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    cache: bool = False
    condition: Optional[str] = None  # Conditional inclusion


class PromptTemplate(BaseModel):
    """Complete prompt template with metadata and messages"""

    metadata: PromptMetadata
    messages: List[MessageTemplate]
    global_variables: Dict[str, Any] = Field(default_factory=dict)
    includes: List[str] = Field(default_factory=list)
    extends: Optional[str] = None


class TemplateProcessor(ABC):
    """Abstract base class for template processors"""

    @abstractmethod
    def render(self, template: str, variables: Dict[str, Any]) -> str:
        """Render template with variables"""

    @abstractmethod
    def validate_template(self, template: str) -> bool:
        """Validate template syntax"""


class StringTemplateProcessor(TemplateProcessor):
    """Python string.Template processor"""

    def render(self, template: str, variables: Dict[str, Any]) -> str:
        try:
            t = Template(template)
            return t.safe_substitute(**variables)
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")

    def validate_template(self, template: str) -> bool:
        try:
            Template(template)
            return True
        except Exception:
            return False


class FormatTemplateProcessor(TemplateProcessor):
    """Python str.format() processor"""

    def render(self, template: str, variables: Dict[str, Any]) -> str:
        try:
            return template.format(**variables)
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")

    def validate_template(self, template: str) -> bool:
        try:
            # Try to format with empty dict to check syntax
            template.format()
            return True
        except (ValueError, KeyError):
            # KeyError is expected if variables are missing
            return True
        except Exception:
            return False


class Jinja2TemplateProcessor(TemplateProcessor):
    """Jinja2 template processor"""

    def __init__(self, template_dir: Optional[str] = None):
        if not HAS_JINJA2:
            raise ImportError("Jinja2 not available. Install with: pip install jinja2")

        if template_dir:
            loader = FileSystemLoader(template_dir)
        else:
            loader = BaseLoader()

        self.env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True)

        # Add custom filters
        self.env.filters["jsonify"] = json.dumps
        self.env.filters["yamlify"] = yaml.dump
        self.env.globals["datetime"] = datetime

    def render(self, template: str, variables: Dict[str, Any]) -> str:
        try:
            t = self.env.from_string(template)
            return t.render(**variables)
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")

    def validate_template(self, template: str) -> bool:
        try:
            self.env.from_string(template)
            return True
        except Exception:
            return False


class PromptLoader:
    """Load prompts from various sources"""

    @staticmethod
    def from_markdown(file_path: Union[str, Path]) -> PromptTemplate:
        """Load prompt from markdown file with YAML frontmatter"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        return PromptLoader.from_markdown_string(content, name=path.stem)

    @staticmethod
    def from_markdown_string(content: str, name: str = "inline") -> PromptTemplate:
        """Load prompt from markdown string with YAML frontmatter"""
        # Check for YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                markdown_content = parts[2].strip()
            else:
                yaml_content = ""
                markdown_content = content
        else:
            yaml_content = ""
            markdown_content = content

        # Parse metadata
        metadata_dict = yaml.safe_load(yaml_content) if yaml_content else {}
        metadata = PromptMetadata(
            name=metadata_dict.get("name", name),
            description=metadata_dict.get("description", ""),
            version=metadata_dict.get("version", "1.0"),
            author=metadata_dict.get("author", ""),
            created_at=metadata_dict.get("created_at"),
            tags=metadata_dict.get("tags", []),
            required_variables=metadata_dict.get("required_variables", []),
            optional_variables=metadata_dict.get("optional_variables", {}),
            template_engine=TemplateEngine(metadata_dict.get("template_engine", "jinja2")),
        )

        # Parse messages from markdown
        messages = PromptLoader._parse_markdown_messages(markdown_content, metadata_dict)

        return PromptTemplate(
            metadata=metadata,
            messages=messages,
            global_variables=metadata_dict.get("global_variables", {}),
            includes=metadata_dict.get("includes", []),
            extends=metadata_dict.get("extends"),
        )

    @staticmethod
    def from_yaml(file_path: Union[str, Path]) -> PromptTemplate:
        """Load prompt from YAML file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return PromptLoader._dict_to_prompt_template(data)

    @staticmethod
    def from_json(file_path: Union[str, Path]) -> PromptTemplate:
        """Load prompt from JSON file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return PromptLoader._dict_to_prompt_template(data)

    @staticmethod
    def from_string(content: str, name: str = "inline", role: str = "system", **metadata_kwargs) -> PromptTemplate:
        """Create prompt from string content"""
        metadata = PromptMetadata(name=name, **metadata_kwargs)

        # Simple string becomes a message with specified role
        messages = [MessageTemplate(role=role, content=content)]

        return PromptTemplate(metadata=metadata, messages=messages)

    @staticmethod
    def _parse_markdown_messages(content: str, metadata: Dict[str, Any]) -> List[MessageTemplate]:
        """Parse message blocks from markdown content"""
        messages = []

        # Look for message blocks: ## role: content or ### role: content
        message_pattern = r"^#+\s*(system|user|assistant)(?:\s*:\s*(.*))?$"
        lines = content.split("\n")

        current_message = None
        current_content = []

        for line in lines:
            match = re.match(message_pattern, line.strip(), re.IGNORECASE)
            if match:
                # Save previous message
                if current_message:
                    current_message.content = "\n".join(current_content).strip()
                    messages.append(current_message)

                # Start new message
                role = match.group(1).lower()
                title_content = match.group(2) or ""
                current_message = MessageTemplate(role=role, content="", cache=metadata.get("cache", False))
                current_content = [title_content] if title_content else []
            else:
                if current_message:
                    current_content.append(line)

        # Save last message
        if current_message:
            current_message.content = "\n".join(current_content).strip()
            messages.append(current_message)

        # If no structured messages found, treat entire content as system message
        if not messages:
            messages = [MessageTemplate(role="system", content=content)]

        return messages

    @staticmethod
    def _dict_to_prompt_template(data: Dict[str, Any]) -> PromptTemplate:
        """Convert dictionary to PromptTemplate"""
        metadata_dict = data.get("metadata", {})
        metadata = PromptMetadata(
            name=metadata_dict.get("name", "unnamed"),
            description=metadata_dict.get("description", ""),
            version=metadata_dict.get("version", "1.0"),
            author=metadata_dict.get("author", ""),
            created_at=metadata_dict.get("created_at"),
            tags=metadata_dict.get("tags", []),
            required_variables=metadata_dict.get("required_variables", []),
            optional_variables=metadata_dict.get("optional_variables", {}),
            template_engine=TemplateEngine(metadata_dict.get("template_engine", "jinja2")),
        )

        messages_data = data.get("messages", [])
        messages = [
            MessageTemplate(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                name=msg.get("name"),
                cache=msg.get("cache", False),
                condition=msg.get("condition"),
            )
            for msg in messages_data
        ]

        return PromptTemplate(
            metadata=metadata,
            messages=messages,
            global_variables=data.get("global_variables", {}),
            includes=data.get("includes", []),
            extends=data.get("extends"),
        )


class PromptManager:
    """Main prompt management system"""

    def __init__(
        self,
        template_dirs: Optional[List[str]] = None,
        default_engine: TemplateEngine = TemplateEngine.JINJA2,
        enable_cache: bool = True,
    ):
        self.template_dirs = [Path(d) for d in (template_dirs or [])]
        self.default_engine = default_engine
        self.enable_cache = enable_cache

        # Template cache
        self._template_cache: Dict[str, PromptTemplate] = {}

        # Processors
        self._processors = {
            TemplateEngine.STRING: StringTemplateProcessor(),
            TemplateEngine.FORMAT: FormatTemplateProcessor(),
        }

        if HAS_JINJA2:
            template_dir = str(self.template_dirs[0]) if self.template_dirs else None
            self._processors[TemplateEngine.JINJA2] = Jinja2TemplateProcessor(template_dir)

        # Custom functions for templates
        self._custom_functions: Dict[str, Callable] = {}

    def register_custom_function(self, name: str, func: Callable):
        """Register custom function for use in templates"""
        self._custom_functions[name] = func

    def load_prompt(
        self,
        file_path: Optional[Union[str, Path]] = None,
        content: Optional[str] = None,
        name: Optional[str] = None,
        **metadata_kwargs,
    ) -> PromptTemplate:
        """Load a prompt template"""

        if file_path:
            path = Path(file_path)

            # Try to find in template directories
            if not path.is_absolute():
                for template_dir in self.template_dirs:
                    candidate = template_dir / path
                    if candidate.exists():
                        path = candidate
                        break

            cache_key = str(path)

            # Check cache
            if self.enable_cache and cache_key in self._template_cache:
                return self._template_cache[cache_key]

            # Load based on file extension
            if path.suffix.lower() == ".md":
                template = PromptLoader.from_markdown(path)
            elif path.suffix.lower() in [".yaml", ".yml"]:
                template = PromptLoader.from_yaml(path)
            elif path.suffix.lower() == ".json":
                template = PromptLoader.from_json(path)
            else:
                # Try markdown first, then treat as plain text
                try:
                    template = PromptLoader.from_markdown(path)
                except:
                    content = path.read_text(encoding="utf-8")
                    template = PromptLoader.from_string(content, name or path.stem)

            # Cache template
            if self.enable_cache:
                self._template_cache[cache_key] = template

            return template

        elif content:
            return PromptLoader.from_string(content, name or "inline", **metadata_kwargs)

        else:
            raise ValueError("Either file_path or content must be provided")

    def render_prompt(
        self, template: Union[PromptTemplate, str, Path], variables: Optional[Dict[str, Any]] = None, **kwargs
    ) -> List[BaseMessage]:
        """Render a prompt template to messages"""

        if not isinstance(template, PromptTemplate):
            template = self.load_prompt(template)

        variables = variables or {}
        variables.update(kwargs)

        # Add global variables and custom functions
        variables.update(template.global_variables)
        variables.update(self._custom_functions)

        # Validate required variables
        missing_vars = set(template.metadata.required_variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        # Add optional variables with defaults
        for var, default in template.metadata.optional_variables.items():
            if var not in variables:
                variables[var] = default

        # Get template processor
        engine = template.metadata.template_engine
        if engine not in self._processors:
            engine = self.default_engine

        processor = self._processors[engine]

        # Render messages
        messages = []
        for msg_template in template.messages:
            # Check condition
            if msg_template.condition:
                try:
                    condition_result = eval(msg_template.condition, {"__builtins__": {}}, variables)
                    if not condition_result:
                        continue
                except Exception as e:
                    print(f"Warning: Condition evaluation failed for message: {e}")
                    continue

            # Render content
            rendered_content = processor.render(msg_template.content, variables)

            # Create message object
            if msg_template.role == "system":
                message = SystemMessage(content=rendered_content, name=msg_template.name, cache=msg_template.cache)
            elif msg_template.role == "user":
                message = UserMessage(content=rendered_content, name=msg_template.name)
            elif msg_template.role == "assistant":
                message = AssistantMessage(content=rendered_content, name=msg_template.name)
            else:
                raise ValueError(f"Unknown role: {msg_template.role}")

            messages.append(message)

        return messages

    def list_templates(self, tag: Optional[str] = None) -> List[str]:
        """List available templates"""
        templates = []

        for template_dir in self.template_dirs:
            if not template_dir.exists():
                continue

            for file_path in template_dir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in [".md", ".yaml", ".yml", ".json"]:
                    try:
                        template = self.load_prompt(file_path)
                        if not tag or tag in template.metadata.tags:
                            templates.append(str(file_path.relative_to(template_dir)))
                    except Exception:
                        continue

        return sorted(templates)

    def get_template_info(self, template_path: Union[str, Path]) -> PromptMetadata:
        """Get metadata for a template"""
        template = self.load_prompt(template_path)
        return template.metadata

    def validate_template(self, template_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a template"""
        result = {"valid": True, "errors": [], "warnings": []}

        try:
            template = self.load_prompt(template_path)

            # Check engine availability
            engine = template.metadata.template_engine
            if engine not in self._processors:
                result["warnings"].append(
                    f"Template engine {engine.value} not available, using {self.default_engine.value}"
                )
                engine = self.default_engine

            processor = self._processors[engine]

            # Validate each message template
            for i, msg_template in enumerate(template.messages):
                if not processor.validate_template(msg_template.content):
                    result["valid"] = False
                    result["errors"].append(f"Invalid template syntax in message {i+1}")

        except Exception as e:
            result["valid"] = False
            result["errors"].append(str(e))

        return result

    def clear_cache(self):
        """Clear template cache"""
        self._template_cache.clear()


# ============================================================================
# Convenience Functions
# ============================================================================


def create_simple_prompt(content: str, role: str = "system", **kwargs) -> List[BaseMessage]:
    """Quick way to create a simple prompt"""
    manager = PromptManager()
    template = PromptLoader.from_string(content, role=role)
    return manager.render_prompt(template, **kwargs)


def load_and_render(file_path: Union[str, Path], **variables) -> List[BaseMessage]:
    """Quick way to load and render a prompt file"""
    manager = PromptManager()
    return manager.render_prompt(file_path, variables)
