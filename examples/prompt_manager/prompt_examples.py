"""
Usage examples for the Prompt Module

This file demonstrates comprehensive usage of the prompt management system,
including various template formats, engines, and advanced features.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Import prompt module
from cogents_core.llm.prompt import PromptLoader, PromptManager, PromptTemplate, TemplateEngine, create_simple_prompt


class PromptExamples:
    """Collection of examples for the prompt module"""

    def __init__(self):
        # Get the examples directory
        self.examples_dir = Path(__file__).parent

        # Initialize prompt manager with examples directory
        self.manager = PromptManager(
            template_dirs=[str(self.examples_dir)], default_engine=TemplateEngine.JINJA2, enable_cache=True
        )

    def basic_usage_examples(self) -> Dict[str, Any]:
        """Basic usage examples"""
        print("=" * 60)
        print("BASIC USAGE EXAMPLES")
        print("=" * 60)

        results = {}

        # 1. Simple string prompt
        print("\n1. Simple String Prompt:")
        simple_messages = create_simple_prompt(
            "You are a helpful assistant for {{ domain }}. Today is {{ date }}.",
            domain="web automation",
            date=datetime.now().strftime("%Y-%m-%d"),
        )

        for msg in simple_messages:
            print(f"   Role: {msg.role}")
            print(f"   Content: {msg.text[:100]}...")

        results["simple_prompt"] = simple_messages

        # 2. Load markdown template
        print("\n2. Loading Markdown Template:")
        try:
            web_search_messages = self.manager.render_prompt(
                "web_search.md",
                search_query="best Python web scraping libraries",
                user_goal="Find reliable tools for web automation",
                max_results=10,
                search_type="technical",
            )

            print(f"   Generated {len(web_search_messages)} messages")
            for i, msg in enumerate(web_search_messages):
                print(f"   Message {i+1}: {msg.role} ({len(msg.text)} chars)")

            results["markdown_template"] = web_search_messages

        except Exception as e:
            print(f"   Error loading markdown template: {e}")

        # 3. Load YAML template
        print("\n3. Loading YAML Template:")
        try:
            extraction_messages = self.manager.render_prompt(
                "data_extraction.yaml",
                target_data="product information",
                output_format="JSON",
                validation_rules=["Price must be numeric", "Name cannot be empty"],
                field_mappings={"item_name": "product_name", "cost": "price"},
            )

            print(f"   Generated {len(extraction_messages)} messages")
            results["yaml_template"] = extraction_messages

        except Exception as e:
            print(f"   Error loading YAML template: {e}")

        # 4. Load JSON template
        print("\n4. Loading JSON Template:")
        try:
            form_messages = self.manager.render_prompt(
                "form_filling.json",
                form_data={"name": "John Doe", "email": "john@example.com", "phone": "+1-555-0123"},
                validation_mode="strict",
                submit_form=False,
            )

            print(f"   Generated {len(form_messages)} messages")
            results["json_template"] = form_messages

        except Exception as e:
            print(f"   Error loading JSON template: {e}")

        return results

    def template_engine_examples(self) -> Dict[str, Any]:
        """Examples using different template engines"""
        print("\n" + "=" * 60)
        print("TEMPLATE ENGINE EXAMPLES")
        print("=" * 60)

        results = {}
        variables = {"name": "John", "balance": 150.75, "currency": "USD"}

        # 1. String Template Engine
        print("\n1. String Template Engine:")
        string_template = PromptLoader.from_string(
            "Hello $name, your $currency balance is $$balance.",
            name="string_example",
            template_engine=TemplateEngine.STRING,
        )

        string_messages = self.manager.render_prompt(string_template, variables)
        print(f"   Result: {string_messages[0].text}")
        results["string_engine"] = string_messages

        # 2. Format Template Engine
        print("\n2. Format Template Engine:")
        format_template = PromptLoader.from_string(
            "Hello {name}, your {currency} balance is ${balance:.2f}.",
            name="format_example",
            template_engine=TemplateEngine.FORMAT,
        )

        format_messages = self.manager.render_prompt(format_template, variables)
        print(f"   Result: {format_messages[0].text}")
        results["format_engine"] = format_messages

        # 3. Jinja2 Template Engine
        print("\n3. Jinja2 Template Engine:")
        jinja_template = PromptLoader.from_string(
            "Hello {{ name }}, your {{ currency }} balance is ${{ '%.2f' | format(balance) }}.",
            name="jinja_example",
            template_engine=TemplateEngine.JINJA2,
        )

        jinja_messages = self.manager.render_prompt(jinja_template, variables)
        print(f"   Result: {jinja_messages[0].text}")
        results["jinja_engine"] = jinja_messages

        return results

    def conditional_logic_examples(self) -> Dict[str, Any]:
        """Examples with conditional logic and dynamic content"""
        print("\n" + "=" * 60)
        print("CONDITIONAL LOGIC EXAMPLES")
        print("=" * 60)

        results = {}

        # Create a template with conditional messages
        conditional_template_content = """
---
name: "Conditional Assistant"
required_variables: ["user_type", "task_complexity"]
optional_variables:
  debug_mode: false
  priority: "medium"
template_engine: "jinja2"
---

## system

You are an AI assistant specialized in {{ user_type }} support.

{% if user_type == "beginner" %}
**Mode:** Beginner-friendly explanations
- Use simple language
- Provide step-by-step instructions
- Include helpful examples
{% elif user_type == "expert" %}
**Mode:** Expert-level responses
- Use technical terminology
- Focus on efficiency
- Assume background knowledge
{% endif %}

Priority level: {{ priority | upper }}
{% if task_complexity == "high" %}
**Note:** High complexity task - take extra care with accuracy.
{% endif %}

## user

{% if user_type == "beginner" %}
Please provide detailed, step-by-step instructions for: {{ task_description }}
{% else %}
Provide a concise solution for: {{ task_description }}
{% endif %}

## assistant
This message appears only in debug mode.
        """

        # Add condition to the assistant message by parsing manually
        conditional_template = PromptLoader.from_markdown_string(conditional_template_content, "conditional_example")

        # Add condition to assistant message
        if conditional_template.messages:
            conditional_template.messages[-1].condition = "debug_mode == True"

        print("\n1. Beginner Mode (Low Complexity):")
        beginner_messages = self.manager.render_prompt(
            conditional_template,
            user_type="beginner",
            task_complexity="low",
            task_description="install Python packages",
            debug_mode=False,
        )

        print(f"   Generated {len(beginner_messages)} messages")
        for msg in beginner_messages:
            print(f"   {msg.role}: {msg.text[:80]}...")

        results["beginner_mode"] = beginner_messages

        print("\n2. Expert Mode (High Complexity, Debug On):")
        expert_messages = self.manager.render_prompt(
            conditional_template,
            user_type="expert",
            task_complexity="high",
            task_description="optimize database queries",
            debug_mode=True,
            priority="high",
        )

        print(f"   Generated {len(expert_messages)} messages")
        for msg in expert_messages:
            print(f"   {msg.role}: {msg.text[:80]}...")

        results["expert_mode"] = expert_messages

        return results

    def custom_functions_examples(self) -> Dict[str, Any]:
        """Examples with custom functions"""
        print("\n" + "=" * 60)
        print("CUSTOM FUNCTIONS EXAMPLES")
        print("=" * 60)

        results = {}

        # Register custom functions
        def format_currency(amount: float, currency: str = "USD") -> str:
            """Format amount as currency"""
            symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
            symbol = symbols.get(currency, currency)
            return f"{symbol}{amount:.2f}"

        def validate_email(email: str) -> bool:
            """Simple email validation"""
            return "@" in email and "." in email.split("@")[1]

        def truncate_text(text: str, max_length: int = 50) -> str:
            """Truncate text to max length"""
            return text[:max_length] + "..." if len(text) > max_length else text

        # Register functions
        self.manager.register_custom_function("format_currency", format_currency)
        self.manager.register_custom_function("validate_email", validate_email)
        self.manager.register_custom_function("truncate", truncate_text)

        # Create template using custom functions
        custom_functions_template = PromptLoader.from_string(
            """
You are processing user data:

**Financial Info:**
- Account balance: {{ format_currency(balance, currency) }}
- Credit limit: {{ format_currency(credit_limit) }}

**Contact Validation:**
- Email: {{ email }} 
{% if validate_email(email) %}
  ✓ Email format is valid
{% else %}
  ✗ Email format is invalid
{% endif %}

**Summary:**
{{ truncate(description, 100) }}
        """,
            name="custom_functions_example",
        )

        print("\n1. Using Custom Functions:")
        custom_messages = self.manager.render_prompt(
            custom_functions_template,
            balance=2500.99,
            credit_limit=5000.00,
            currency="EUR",
            email="user@example.com",
            description="This is a very long description that will be truncated to demonstrate the custom truncation function working properly in templates.",
        )

        print(f"   Result:\n{custom_messages[0].text}")
        results["custom_functions"] = custom_messages

        return results

    def template_validation_examples(self) -> Dict[str, Any]:
        """Template validation examples"""
        print("\n" + "=" * 60)
        print("TEMPLATE VALIDATION EXAMPLES")
        print("=" * 60)

        results = {}

        # Validate existing templates
        templates_to_validate = ["web_search.md", "data_extraction.yaml", "form_filling.json"]

        for template_name in templates_to_validate:
            try:
                validation_result = self.manager.validate_template(template_name)
                print(f"\n{template_name}:")
                print(f"   Valid: {validation_result['valid']}")

                if validation_result["errors"]:
                    print("   Errors:")
                    for error in validation_result["errors"]:
                        print(f"     - {error}")

                if validation_result["warnings"]:
                    print("   Warnings:")
                    for warning in validation_result["warnings"]:
                        print(f"     - {warning}")

                results[template_name] = validation_result

            except Exception as e:
                print(f"   Error validating {template_name}: {e}")

        # Test invalid template
        print(f"\nInvalid Template Test:")
        invalid_template = PromptLoader.from_string(
            "Hello {{ name }}, your balance is {{ invalid_syntax", name="invalid_example"
        )

        # Create temporary file for validation
        temp_file = self.examples_dir / "temp_invalid.json"
        try:
            with open(temp_file, "w") as f:
                json.dump(
                    {"metadata": {"name": "invalid"}, "messages": [{"role": "system", "content": "{{ unclosed_tag"}]}, f
                )

            invalid_result = self.manager.validate_template(temp_file)
            print(f"   Valid: {invalid_result['valid']}")
            print(f"   Errors: {invalid_result['errors']}")
            results["invalid_template"] = invalid_result

        finally:
            if temp_file.exists():
                temp_file.unlink()

        return results

    def template_management_examples(self) -> Dict[str, Any]:
        """Template management and metadata examples"""
        print("\n" + "=" * 60)
        print("TEMPLATE MANAGEMENT EXAMPLES")
        print("=" * 60)

        results = {}

        # List all templates
        print("\n1. Available Templates:")
        all_templates = self.manager.list_templates()
        for template in all_templates:
            print(f"   - {template}")
        results["all_templates"] = all_templates

        # List templates by tag
        print("\n2. Templates by Tag:")
        web_templates = self.manager.list_templates(tag="web")
        extraction_templates = self.manager.list_templates(tag="extraction")

        print(f"   Web templates: {web_templates}")
        print(f"   Extraction templates: {extraction_templates}")

        results["web_templates"] = web_templates
        results["extraction_templates"] = extraction_templates

        # Get template metadata
        print("\n3. Template Metadata:")
        for template_name in all_templates[:2]:  # Show first 2 templates
            try:
                metadata = self.manager.get_template_info(template_name)
                print(f"   {template_name}:")
                print(f"     Name: {metadata.name}")
                print(f"     Description: {metadata.description}")
                print(f"     Version: {metadata.version}")
                print(f"     Tags: {metadata.tags}")
                print(f"     Required vars: {metadata.required_variables}")
                print(f"     Optional vars: {list(metadata.optional_variables.keys())}")

                results[f"metadata_{template_name}"] = metadata

            except Exception as e:
                print(f"     Error getting metadata: {e}")

        return results

    def advanced_features_examples(self) -> Dict[str, Any]:
        """Advanced features examples"""
        print("\n" + "=" * 60)
        print("ADVANCED FEATURES EXAMPLES")
        print("=" * 60)

        results = {}

        # 1. Multi-message templates
        print("\n1. Multi-Message Template:")
        multi_message_template = """
---
name: "Conversation Starter"
required_variables: ["topic", "user_name"]
optional_variables:
  conversation_style: "friendly"
  expertise_level: "intermediate"
---

## system

You are starting a conversation about {{ topic }}.
Style: {{ conversation_style | title }}
User expertise: {{ expertise_level }}

## user

Hi, I'm {{ user_name }}. I'd like to learn about {{ topic }}.

## assistant

Hello {{ user_name }}! I'd be happy to help you learn about {{ topic }}.

{% if expertise_level == "beginner" %}
Let's start with the basics and work our way up.
{% elif expertise_level == "advanced" %}
I assume you have some background, so we can dive into more complex aspects.
{% else %}
We can adjust the complexity as we go based on your comfort level.
{% endif %}

What specific aspect interests you most?
        """

        multi_template = PromptLoader.from_markdown_string(multi_message_template)
        multi_messages = self.manager.render_prompt(
            multi_template,
            topic="machine learning",
            user_name="Alex",
            conversation_style="professional",
            expertise_level="beginner",
        )

        print(f"   Generated conversation with {len(multi_messages)} messages:")
        for i, msg in enumerate(multi_messages):
            print(f"     Message {i+1} ({msg.role}): {msg.text[:50]}...")

        results["multi_message"] = multi_messages

        # 2. Template inheritance simulation
        print("\n2. Template with Global Variables:")
        global_vars_template = """
---
name: "Report Generator"
required_variables: ["report_type", "data"]
global_variables:
  company_name: "Acme Corp"
  report_date: "{{ datetime.now().strftime('%Y-%m-%d') }}"
  footer: "Generated by {{ company_name }} AI System"
---

## system

Generate a {{ report_type }} report for {{ company_name }}.

**Report Details:**
- Date: {{ report_date }}
- Type: {{ report_type | title }}
- Data points: {{ data | length }}

{{ footer }}

## user

Please analyze the following data: {{ data | truncate(100) }}
        """

        global_template = PromptLoader.from_markdown_string(global_vars_template)
        global_messages = self.manager.render_prompt(
            global_template, report_type="sales", data=["Q1: $50k", "Q2: $75k", "Q3: $60k", "Q4: $90k"]
        )

        print(f"   Report template result: {len(global_messages)} messages")
        print(f"   System message: {global_messages[0].text[:100]}...")

        results["global_variables"] = global_messages

        # 3. Caching demonstration
        print("\n3. Caching Performance:")
        import time

        # First load (cache miss)
        start_time = time.time()
        self.manager.render_prompt("web_search.md", search_query="test", user_goal="test")
        first_load_time = time.time() - start_time

        # Second load (cache hit)
        start_time = time.time()
        self.manager.render_prompt("web_search.md", search_query="test2", user_goal="test2")
        second_load_time = time.time() - start_time

        print(f"   First load (cache miss): {first_load_time:.4f}s")
        print(f"   Second load (cache hit): {second_load_time:.4f}s")
        print(f"   Speedup: {first_load_time/second_load_time:.1f}x")

        results["caching_performance"] = {
            "first_load": first_load_time,
            "second_load": second_load_time,
            "speedup": first_load_time / second_load_time,
        }

        return results

    def error_handling_examples(self) -> Dict[str, Any]:
        """Error handling and edge cases examples"""
        print("\n" + "=" * 60)
        print("ERROR HANDLING EXAMPLES")
        print("=" * 60)

        results = {}

        # 1. Missing required variables
        print("\n1. Missing Required Variables:")
        try:
            self.manager.render_prompt("web_search.md", search_query="test")  # Missing user_goal
            print("   Unexpected success!")
        except ValueError as e:
            print(f"   Expected error caught: {e}")
            results["missing_vars_error"] = str(e)

        # 2. Invalid template file
        print("\n2. Invalid Template File:")
        try:
            self.manager.render_prompt("nonexistent_template.md")
            print("   Unexpected success!")
        except FileNotFoundError:
            print(f"   Expected error caught: File not found")
            results["file_not_found"] = True

        # 3. Template with syntax error
        print("\n3. Template Syntax Error Handling:")
        try:
            broken_template = PromptLoader.from_string(
                "Hello {{ name }, your balance is {{ amount", name="broken_example"  # Broken syntax
            )
            self.manager.render_prompt(broken_template, name="test", amount=100)
            print("   Unexpected success!")
        except ValueError:
            print(f"   Expected error caught: Template syntax error")
            results["syntax_error"] = True

        # 4. Graceful fallback
        print("\n4. Graceful Fallback:")
        try:
            # Template with optional variables
            fallback_template = PromptLoader.from_string(
                "Hello {{ name }}{% if title %}, {{ title }}{% endif %}!", name="fallback_example"
            )
            fallback_messages = self.manager.render_prompt(
                fallback_template, name="User"  # title is optional and missing
            )
            print(f"   Graceful handling: {fallback_messages[0].text}")
            results["graceful_fallback"] = fallback_messages
        except Exception as e:
            print(f"   Unexpected error: {e}")

        return results

    def integration_examples(self) -> Dict[str, Any]:
        """Integration examples with external systems"""
        print("\n" + "=" * 60)
        print("INTEGRATION EXAMPLES")
        print("=" * 60)

        results = {}

        # 1. Dynamic template creation
        print("\n1. Dynamic Template Creation:")

        def create_dynamic_template(task_type: str, complexity: str) -> PromptTemplate:
            """Create templates dynamically based on parameters"""

            if task_type == "analysis":
                content = """
You are a data analyst. Analyze the provided data with {{ complexity }} detail.

{% if complexity == "high" %}
Provide comprehensive statistical analysis, visualizations recommendations, and actionable insights.
{% else %}
Provide a summary analysis with key findings.
{% endif %}
                """
            elif task_type == "creative":
                content = """
You are a creative assistant. Help with {{ task_name }} using {{ complexity }} creativity.

{% if complexity == "high" %}
Think outside the box, explore unconventional approaches, and provide multiple creative alternatives.
{% else %}
Provide straightforward creative suggestions.
{% endif %}
                """
            else:
                content = "You are a general assistant. Help with: {{ task_name }}"

            return PromptLoader.from_string(
                content,
                name=f"dynamic_{task_type}_{complexity}",
                required_variables=["task_name"] if task_type != "analysis" else [],
            )

        # Create and use dynamic templates
        analysis_template = create_dynamic_template("analysis", "high")
        analysis_messages = self.manager.render_prompt(analysis_template, complexity="high")

        creative_template = create_dynamic_template("creative", "low")
        creative_messages = self.manager.render_prompt(creative_template, task_name="logo design", complexity="low")

        print(f"   Analysis template: {len(analysis_messages)} messages")
        print(f"   Creative template: {len(creative_messages)} messages")

        results["dynamic_templates"] = {"analysis": analysis_messages, "creative": creative_messages}

        # 2. Message serialization
        print("\n2. Message Serialization:")
        messages = create_simple_prompt("Test message for serialization", role="system")

        # Convert to dict for JSON serialization
        serialized = []
        for msg in messages:
            serialized.append({"role": msg.role, "content": msg.text, "name": msg.name, "cache": msg.cache})

        json_str = json.dumps(serialized, indent=2)
        print(f"   Serialized messages:\n{json_str}")
        results["serialization"] = serialized

        return results

    def run_all_examples(self) -> Dict[str, Any]:
        """Run all examples and return comprehensive results"""
        print("🚀 Running Comprehensive Prompt Module Examples")
        print("=" * 80)

        all_results = {}

        try:
            all_results["basic_usage"] = self.basic_usage_examples()
            all_results["template_engines"] = self.template_engine_examples()
            all_results["conditional_logic"] = self.conditional_logic_examples()
            all_results["custom_functions"] = self.custom_functions_examples()
            all_results["template_validation"] = self.template_validation_examples()
            all_results["template_management"] = self.template_management_examples()
            all_results["advanced_features"] = self.advanced_features_examples()
            all_results["error_handling"] = self.error_handling_examples()
            all_results["integration"] = self.integration_examples()

            print("\n" + "=" * 80)
            print("✅ All examples completed successfully!")
            print("=" * 80)

        except Exception as e:
            print(f"\n❌ Error running examples: {e}")
            import traceback

            traceback.print_exc()

        return all_results


def main():
    """Main function to run examples"""
    examples = PromptExamples()
    results = examples.run_all_examples()

    # Save results to file for reference
    results_file = Path(__file__).parent / "example_results.json"
    try:
        with open(results_file, "w") as f:
            # Convert non-serializable objects to strings
            serializable_results = {}
            for category, data in results.items():
                if isinstance(data, dict):
                    serializable_results[category] = {}
                    for key, value in data.items():
                        if hasattr(value, "model_dump"):
                            # Pydantic model
                            serializable_results[category][key] = value.model_dump()
                        elif isinstance(value, list) and value and hasattr(value[0], "text"):
                            # List of message objects
                            serializable_results[category][key] = [
                                {"role": msg.role, "content": msg.text} for msg in value
                            ]
                        else:
                            serializable_results[category][key] = str(value)
                else:
                    serializable_results[category] = str(data)

            json.dump(serializable_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

    return results


if __name__ == "__main__":
    main()
