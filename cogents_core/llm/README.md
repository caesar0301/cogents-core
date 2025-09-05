# Prompt Module Documentation

A comprehensive and extensible prompt management system for handling various prompt templates, dynamic variable injection, and multi-format support.

## Features

- **Multiple Template Formats**: Support for Markdown, YAML, JSON, and inline strings
- **Template Engines**: String templates, Python format strings, and Jinja2
- **Dynamic Variables**: Inject variables with conditional logic and custom functions
- **Message Types**: Support for system, user, and assistant message roles
- **Caching**: Template caching for improved performance
- **Validation**: Template syntax and variable validation
- **Extensible**: Easy to add custom functions and processors

## Quick Start

```python
from cogents_core.llm.prompt import PromptManager, create_simple_prompt

# Quick simple prompt
messages = create_simple_prompt(
    "You are a helpful assistant for {{ domain }}.",
    domain="web automation"
)

# Load from file
manager = PromptManager(template_dirs=["prompt/templates"])
messages = manager.render_prompt(
    "web_search.md",
    search_query="Python tutorials",
    user_goal="Learn Python basics"
)
```

## Template Formats

### Markdown with YAML Frontmatter

```markdown
---
name: "My Template"
description: "A sample template"
required_variables: ["user_name"]
optional_variables:
  greeting: "Hello"
template_engine: "jinja2"
---

## system
{{ greeting }} {{ user_name }}! How can I help you today?

## user
I need assistance with my task.
```

### YAML Format

```yaml
metadata:
  name: "YAML Template"
  required_variables: ["task"]
  template_engine: "jinja2"

messages:
  - role: "system"
    content: "You will help with: {{ task }}"
    cache: true
  - role: "user" 
    content: "Let's get started!"
```

### JSON Format

```json
{
  "metadata": {
    "name": "JSON Template",
    "required_variables": ["objective"]
  },
  "messages": [
    {
      "role": "system",
      "content": "Your objective: {{ objective }}"
    }
  ]
}
```

## Template Engines

### Jinja2 (Recommended)
- Full templating power with conditions, loops, filters
- Custom functions and filters supported
- Template inheritance and includes

```jinja2
{% if user_type == "admin" %}
You have administrative privileges.
{% endif %}

{% for item in items %}
- {{ item.name }}: {{ item.value }}
{% endfor %}
```

### Python Format Strings
- Simple variable substitution
- Number formatting support

```python
"Hello {name}, your score is {score:.2f}"
```

### String Templates
- Safe substitution with $ syntax
- Prevents code injection

```python
"Hello $name, welcome to $platform"
```

## Advanced Features

### Conditional Messages

```yaml
messages:
  - role: "system"
    content: "Base system message"
  - role: "user"
```

