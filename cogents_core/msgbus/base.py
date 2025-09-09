"""Base watchdog class for general event monitoring components.

This is a generic watchdog class that can be used with any event processor that
implements the EventProcessor protocol. Derived from browser-use BaseWatchdog.
"""

import time
from collections.abc import Iterable
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from bubus import BaseEvent, EventBus
from pydantic import BaseModel, ConfigDict, Field, model_validator

from cogents_core.utils.logging import color_text

# Generic type for the event processor
TEventProcessor = TypeVar("TEventProcessor", bound="EventProcessor")


class EventProcessor(Protocol):
    """Protocol defining the interface for event processors."""

    @property
    def event_bus(self) -> EventBus:
        """Get the event bus instance."""
        ...

    @property
    def logger(self):
        """Get the logger instance."""
        ...


class BaseWatchdog(BaseModel, Generic[TEventProcessor]):
    """Base class for all event watchdogs.

    Watchdogs monitor events and emit new events based on changes.
    They automatically register event handlers based on method names.

    Handler methods should be named: on_EventTypeName(self, event: EventTypeName)

    Generic type TEventProcessor allows you to specify the type of event processor
    this watchdog works with (e.g., DatabaseSession, etc.)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # allow non-serializable objects like EventBus/EventProcessor in fields
        extra="forbid",  # dont allow implicit class/instance state, everything must be a properly typed Field or PrivateAttr
        validate_assignment=False,  # avoid re-triggering  __init__ / validators on values on every assignment
        revalidate_instances="never",  # avoid re-triggering __init__ / validators and erasing private attrs
    )

    # Core dependencies
    event_bus: EventBus = Field()
    event_processor: Any = Field()  # Use Any to avoid Pydantic validation issues with generic types

    @model_validator(mode="after")
    def validate_event_bus_consistency(self) -> "BaseWatchdog":
        """Validate that event_processor has the same event_bus instance as the watchdog.

        This prevents the architectural issue where events are dispatched to one bus
        but handlers listen on a different bus, causing infinite hangs.
        """
        if not hasattr(self.event_processor, "event_bus"):
            raise ValueError(
                f"EventProcessor {type(self.event_processor).__name__} must have an 'event_bus' attribute. "
                f"Ensure your event processor implements the EventProcessor protocol correctly."
            )

        if not hasattr(self.event_processor, "logger"):
            raise ValueError(
                f"EventProcessor {type(self.event_processor).__name__} must have an 'logger' attribute. "
                f"Ensure your event processor implements the EventProcessor protocol correctly."
            )

        processor_bus = self.event_processor.event_bus
        watchdog_bus = self.event_bus

        if processor_bus is not watchdog_bus:
            raise ValueError(
                f"EventProcessor.event_bus and BaseWatchdog.event_bus must be the same instance! "
                f"Found different instances: {type(processor_bus)} vs {type(watchdog_bus)}. "
                f"This causes events to be dispatched to one bus while handlers listen on another, "
                f"resulting in infinite hangs. Ensure both use the same EventBus instance."
            )

        return self

    # Class variables to statically define the list of events relevant to each watchdog
    # (not enforced, just to make it easier to understand the code and debug watchdogs at runtime)
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = []  # Events this watchdog listens to
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []  # Events this watchdog emits

    # Shared state that other watchdogs might need to access should not be defined on EventProcessor, not here!
    # Shared helper methods needed by other watchdogs should be defined on EventProcessor, not here!
    # Alternatively, expose some events on the watchdog to allow access to state/helpers via event_bus system.

    # Private state internal to the watchdog can be defined like this on BaseWatchdog subclasses:
    # _cache: dict[str, bytes] = PrivateAttr(default_factory=dict)
    # _watcher_task: asyncio.Task | None = PrivateAttr(default=None)
    # _download_tasks: WeakSet[asyncio.Task] = PrivateAttr(default_factory=WeakSet)
    # ...

    @property
    def logger(self):
        """Get the logger from the event processor."""
        return self.event_processor.logger

    @staticmethod
    def attach_handler_to_processor(
        event_processor: EventProcessor, event_class: type[BaseEvent[Any]], handler
    ) -> None:
        """Attach a single event handler to an event processor.

        Args:
            event_processor: The event processor to attach to
            event_class: The event class to listen for
            handler: The handler method (must start with 'on_' and end with event type)
        """
        event_bus = event_processor.event_bus

        # Validate handler naming convention
        assert hasattr(handler, "__name__"), "Handler must have a __name__ attribute"
        assert handler.__name__.startswith("on_"), f'Handler {handler.__name__} must start with "on_"'
        assert handler.__name__.endswith(
            event_class.__name__
        ), f"Handler {handler.__name__} must end with event type {event_class.__name__}"

        # Get the watchdog instance if this is a bound method
        watchdog_instance = getattr(handler, "__self__", None)
        watchdog_class_name = watchdog_instance.__class__.__name__ if watchdog_instance else "Unknown"

        # Create a wrapper function with unique name to avoid duplicate handler warnings
        # Capture handler by value to avoid closure issues
        def make_unique_handler(actual_handler):
            async def unique_handler(event):
                # Safe event history access - avoid hanging during registration
                try:
                    parent_event = event_bus.event_history.get(event.event_parent_id) if event.event_parent_id else None
                    grandparent_event = (
                        event_bus.event_history.get(parent_event.event_parent_id)
                        if parent_event and parent_event.event_parent_id
                        else None
                    )
                    if parent_event:
                        parent_info = f"{color_text('↲  triggered by', 'yellow')} {color_text(f'on_{parent_event.event_type}#{parent_event.event_id[-4:]}', 'cyan')}"
                        return_info = f"⤴  {color_text('returned to', 'green')} {color_text(f'on_{parent_event.event_type}#{parent_event.event_id[-4:]}', 'cyan')}"
                    else:
                        parent_info = color_text("👈 by EventProcessor", "magenta")
                        return_info = (
                            f"👉 {color_text('returned to', 'green')} {color_text('EventProcessor', 'magenta')}"
                        )

                    grandparent_info = ""
                    if parent_event and grandparent_event:
                        grandparent_info = f" {color_text('↲  under', 'yellow')} {color_text(f'{grandparent_event.event_type}#{grandparent_event.event_id[-4:]}', 'cyan')}"
                except Exception:
                    # Fallback logging if event history access fails
                    parent_info = color_text("👈 by EventProcessor", "magenta")
                    return_info = f"👉 {color_text('returned to', 'green')} {color_text('EventProcessor', 'magenta')}"
                    grandparent_info = ""

                event_str = f"#{event.event_id[-4:]}" if hasattr(event, "event_id") and event.event_id else ""
                time_start = time.time()
                watchdog_and_handler_str = f"[{watchdog_class_name}.{actual_handler.__name__}({event_str})]".ljust(54)
                event_processor.logger.debug(
                    f"{color_text('🚌', 'cyan')} {watchdog_and_handler_str} ⏳ Starting...       {parent_info}{grandparent_info}"
                )

                try:
                    # **EXECUTE THE EVENT HANDLER FUNCTION**
                    result = await actual_handler(event)

                    if isinstance(result, Exception):
                        raise result

                    # just for debug logging, not used for anything else
                    time_end = time.time()
                    time_elapsed = time_end - time_start
                    result_summary = (
                        "" if result is None else f" ➡️ {color_text(f'<{type(result).__name__}>', 'magenta')}"
                    )
                    event_processor.logger.debug(
                        f"{color_text('🚌', 'green')} {watchdog_and_handler_str} ✅ Succeeded ({time_elapsed:.2f}s){result_summary} {return_info}"
                    )
                    return result
                except Exception as e:
                    time_end = time.time()
                    time_elapsed = time_end - time_start
                    event_processor.logger.error(
                        f"{color_text('🚌', 'red')} {watchdog_and_handler_str} ❌ Failed ({time_elapsed:.2f}s): {type(e).__name__}: {e}"
                    )

                    # Attempt to handle errors - subclasses can override this method
                    try:
                        await watchdog_instance._handle_handler_error(e, event, actual_handler)
                    except Exception as sub_error:
                        event_processor.logger.error(
                            f"{color_text('🚌', 'red')} {watchdog_and_handler_str} ❌ Error handling failed: {type(sub_error).__name__}: {sub_error}"
                        )
                        raise

                    raise

            return unique_handler

        unique_handler = make_unique_handler(handler)
        unique_handler.__name__ = f"{watchdog_class_name}.{handler.__name__}"

        # Check if this handler is already registered - throw error if duplicate
        existing_handlers = event_bus.handlers.get(event_class.__name__, [])
        handler_names = [getattr(h, "__name__", str(h)) for h in existing_handlers]

        if unique_handler.__name__ in handler_names:
            raise RuntimeError(
                f"[{watchdog_class_name}] Duplicate handler registration attempted! "
                f"Handler {unique_handler.__name__} is already registered for {event_class.__name__}. "
                f"This likely means attach_to_processor() was called multiple times."
            )

        event_bus.on(event_class, unique_handler)

    async def _handle_handler_error(self, error: Exception, event: BaseEvent[Any], handler) -> None:
        """Handle errors that occur in event handlers.

        Subclasses can override this method to implement custom error handling logic.
        Default implementation does nothing.

        Args:
            error: The exception that occurred
            event: The event that was being processed
            handler: The handler method that failed
        """

    def attach_to_processor(self) -> None:
        """Attach watchdog to its event processor and start monitoring.

        This method handles event listener registration. The watchdog is already
        bound to an event processor via self.event_processor from initialization.
        """
        # Register event handlers automatically based on method names
        assert self.event_processor is not None, "Event processor not initialized"

        # Create efficient event class lookup
        event_class_map = {}

        # Primary strategy: Use LISTENS_TO for efficient event class discovery
        if self.LISTENS_TO:
            event_class_map = {event_class.__name__: event_class for event_class in self.LISTENS_TO}
            self.logger.debug(
                f"[{self.__class__.__name__}] Using LISTENS_TO for event discovery: {list(event_class_map.keys())}"
            )
        else:
            # Safe fallback strategy: Try to discover event classes from event bus event registry
            # This is more reliable than trying to extract from handler annotations during registration
            try:
                # Check if the event bus has an event registry or similar mechanism
                if hasattr(self.event_bus, "_event_types"):
                    event_class_map = {cls.__name__: cls for cls in self.event_bus._event_types}
                elif hasattr(self.event_bus, "event_registry"):
                    event_class_map = {name: cls for name, cls in self.event_bus.event_registry.items()}
                else:
                    # Last resort: try to extract from existing handlers (but do it safely)
                    for event_name, handlers in self.event_bus.handlers.items():
                        if handlers and hasattr(handlers[0], "__annotations__"):
                            # Get the event class from handler's first parameter annotation
                            annotations = handlers[0].__annotations__
                            if "event" in annotations:
                                event_class = annotations["event"]
                                if isinstance(event_class, type) and issubclass(event_class, BaseEvent):
                                    event_class_map[event_name] = event_class

                if event_class_map:
                    self.logger.debug(
                        f"[{self.__class__.__name__}] Discovered event classes: {list(event_class_map.keys())}"
                    )
                else:
                    self.logger.warning(
                        f"[{self.__class__.__name__}] No event classes discovered. Define LISTENS_TO for better performance."
                    )
            except Exception as e:
                self.logger.warning(f"[{self.__class__.__name__}] Failed to discover event classes: {e}")

        # Find all handler methods (on_EventName) and register them efficiently
        registered_events = set()
        handler_methods = []

        # Collect handler methods first
        for method_name in dir(self):
            if method_name.startswith("on_") and callable(getattr(self, method_name)):
                handler_methods.append(method_name)

        # Process each handler method
        for method_name in handler_methods:
            # Extract event name from method name (on_EventName -> EventName)
            event_name = method_name[3:]  # Remove 'on_' prefix

            # Look up event class efficiently
            event_class = event_class_map.get(event_name)

            if event_class:
                # ASSERTION: If LISTENS_TO is defined, enforce it
                if self.LISTENS_TO:
                    assert event_class in self.LISTENS_TO, (
                        f"[{self.__class__.__name__}] Handler {method_name} listens to {event_name} "
                        f"but {event_name} is not declared in LISTENS_TO: {[e.__name__ for e in self.LISTENS_TO]}"
                    )

                handler = getattr(self, method_name)

                # Use the static helper to attach the handler
                self.attach_handler_to_processor(self.event_processor, event_class, handler)
                registered_events.add(event_class)

                self.logger.debug(f"[{self.__class__.__name__}] Registered handler {method_name} for {event_name}")
            else:
                # Better error message for missing event classes
                if self.LISTENS_TO:
                    available_events = [e.__name__ for e in self.LISTENS_TO]
                    self.logger.warning(
                        f"[{self.__class__.__name__}] Handler {method_name} references unknown event '{event_name}'. "
                        f"Available events in LISTENS_TO: {available_events}"
                    )
                else:
                    self.logger.warning(
                        f"[{self.__class__.__name__}] Handler {method_name} references unknown event '{event_name}'. "
                        f"Consider defining LISTENS_TO class variable for better event discovery."
                    )

        # ASSERTION: If LISTENS_TO is defined, ensure all declared events have handlers
        if self.LISTENS_TO:
            missing_handlers = set(self.LISTENS_TO) - registered_events
            if missing_handlers:
                missing_names = [e.__name__ for e in missing_handlers]
                missing_method_names = [f"on_{name}" for name in missing_names]
                self.logger.warning(
                    f"[{self.__class__.__name__}] LISTENS_TO declares {missing_names} "
                    f'but no handlers found (missing {", ".join(missing_method_names)} methods)'
                )

        self.logger.info(f"[{self.__class__.__name__}] Successfully registered {len(registered_events)} event handlers")

    def emit_event(self, event: BaseEvent[Any]) -> None:
        """Emit an event to the event bus.

        Args:
            event: The event to emit
        """
        if self.EMITS:
            event_type = type(event)
            assert event_type in self.EMITS, (
                f"[{self.__class__.__name__}] Attempting to emit {event_type.__name__} "
                f"but it is not declared in EMITS: {[e.__name__ for e in self.EMITS]}"
            )

        self.event_bus.dispatch(event)

    def __del__(self) -> None:
        """Clean up any running tasks during garbage collection."""

        # A BIT OF MAGIC: Cancel any private attributes that look like asyncio tasks
        try:
            for attr_name in dir(self):
                # e.g. _watcher_task = asyncio.Task
                if attr_name.startswith("_") and attr_name.endswith("_task"):
                    try:
                        task = getattr(self, attr_name)
                        if hasattr(task, "cancel") and callable(task.cancel) and not task.done():
                            task.cancel()
                            # self.logger.debug(f'[{self.__class__.__name__}] Cancelled {attr_name} during cleanup')
                    except Exception:
                        pass  # Ignore errors during cleanup

                # e.g. _download_tasks = WeakSet[asyncio.Task] or list[asyncio.Task]
                if (
                    attr_name.startswith("_")
                    and attr_name.endswith("_tasks")
                    and isinstance(getattr(self, attr_name), Iterable)
                ):
                    for task in getattr(self, attr_name):
                        try:
                            if hasattr(task, "cancel") and callable(task.cancel) and not task.done():
                                task.cancel()
                                # self.logger.debug(f'[{self.__class__.__name__}] Cancelled {attr_name} during cleanup')
                        except Exception:
                            pass  # Ignore errors during cleanup
        except Exception as e:
            # Use a basic logger if available, otherwise ignore
            try:
                if hasattr(self, "logger"):
                    self.logger.error(
                        f"⚠️ Error during {self.__class__.__name__} garbage collection __del__(): {type(e)}: {e}"
                    )
            except Exception:
                pass  # Ignore errors during cleanup
