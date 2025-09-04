# Generic Event-Watchdog Framework

A flexible, generic framework for building event-driven monitoring systems using the bubus library.

## Overview

The `BaseWatchdog` class provides a foundation for creating watchdogs that can monitor events from any type of event processor (databases, APIs, browsers, etc.) and emit new events based on their observations.

## Key Features

- **Generic Event Processor Support**: Works with any type of event processor that implements the `EventProcessor` protocol
- **Automatic Event Handler Registration**: Automatically discovers and registers event handlers based on method naming conventions
- **Event Validation**: Enforces `LISTENS_TO` and `EMITS` declarations for better code clarity and debugging
- **Comprehensive Logging**: Built-in colored logging with event tracing and performance metrics
- **Error Handling**: Extensible error handling with the ability to override default behavior
- **Task Management**: Automatic cleanup of asyncio tasks during garbage collection

## Core Components

### EventProcessor Protocol

```python
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
```

### BaseWatchdog Class

```python
class BaseWatchdog(BaseModel, Generic[TEventProcessor]):
    """Base class for all event watchdogs."""
    
    # Core dependencies
    event_bus: EventBus = Field()
    event_processor: TEventProcessor = Field()
    
    # Event declarations
    LISTENS_TO: ClassVar[list[type[BaseEvent[Any]]]] = []
    EMITS: ClassVar[list[type[BaseEvent[Any]]]] = []
```

## Usage

### 1. Define Your Events

```python
from bubus import BaseEvent
from pydantic import Field

class UserLoginEvent(BaseEvent[Any]):
    user_id: str = Field()
    timestamp: float = Field()

class SecurityAlertEvent(BaseEvent[Any]):
    alert_type: str = Field()
    user_id: str = Field()
    severity: str = Field()
```

### 2. Create Your Event Processor

```python
class DatabaseSession:
    def __init__(self, name: str):
        self.name = name
        self.event_bus = EventBus()
        self.logger = logging.getLogger(f"DatabaseSession.{name}")
    
    @property
    def event_bus(self) -> EventBus:
        return self._event_bus
    
    @property
    def logger(self):
        return self._logger
```

### 3. Create Your Watchdog

```python
class UserActivityWatchdog(BaseWatchdog[DatabaseSession]):
    """Watchdog that monitors user activity and emits security alerts."""
    
    # Declare which events this watchdog listens to
    LISTENS_TO = [UserLoginEvent, UserLogoutEvent]
    
    # Declare which events this watchdog emits
    EMITS = [SecurityAlertEvent]
    
    async def on_UserLoginEvent(self, event: UserLoginEvent):
        """Handle user login events."""
        # Your logic here
        if suspicious_activity_detected:
            alert = SecurityAlertEvent(...)
            self.emit_event(alert)
    
    async def on_UserLogoutEvent(self, event: UserLogoutEvent):
        """Handle user logout events."""
        # Your logic here
        pass
```

### 4. Wire Everything Together

```python
# Create instances
event_bus = EventBus()
db_session = DatabaseSession("main_db")
user_watchdog = UserActivityWatchdog(event_bus=event_bus, event_processor=db_session)

# Attach watchdog to start monitoring
user_watchdog.attach_to_processor()

# Start emitting events
db_session.login_user("user123")
```

## Event Handler Naming Convention

Event handlers must follow this naming pattern:
- **Format**: `on_EventTypeName(self, event: EventTypeName)`
- **Example**: `on_UserLoginEvent(self, event: UserLoginEvent)`

The framework automatically discovers these methods and registers them as event handlers.

## Event Validation

### LISTENS_TO
Declare which events your watchdog listens to:
```python
LISTENS_TO = [UserLoginEvent, UserLogoutEvent]
```

### EMITS
Declare which events your watchdog emits:
```python
EMITS = [SecurityAlertEvent, PerformanceAlertEvent]
```

The framework validates these declarations and provides helpful error messages if they don't match your implementation.

## Error Handling

Override the `_handle_handler_error` method to implement custom error handling:

```python
async def _handle_handler_error(self, error: Exception, event: BaseEvent[Any], handler) -> None:
    """Custom error handling logic."""
    if isinstance(error, ConnectionError):
        # Attempt to reconnect
        await self._reconnect()
    else:
        # Log and re-raise
        self.logger.error(f"Handler {handler.__name__} failed: {error}")
        raise
```

## Logging and Debugging

The framework provides comprehensive logging with:
- Event execution timing
- Event hierarchy tracing
- Color-coded output for different log levels
- Performance metrics

Example log output:
```
🚌 [UserActivityWatchdog.on_UserLoginEvent(#abcd)] ⏳ Starting...      👈 by EventProcessor
🚌 [UserActivityWatchdog.on_UserLoginEvent(#abcd)] ✅ Succeeded (0.05s) ➡️ <None> 👉 returned to  EventProcessor
```

## Advanced Features

### Custom Event Emission

```python
def emit_event(self, event: BaseEvent[Any]) -> None:
    """Emit an event with validation."""
    # The framework validates against EMITS declaration
    super().emit_event(event)
```

### Task Management

The framework automatically manages asyncio tasks:
- Cancels tasks during cleanup
- Supports both single tasks (`_watcher_task`) and collections (`_download_tasks`)
- Graceful error handling during cleanup

## Best Practices

1. **Always declare LISTENS_TO and EMITS**: This improves code clarity and enables validation
2. **Use descriptive event names**: Make your events self-documenting
3. **Keep handlers focused**: Each handler should do one thing well
4. **Handle errors gracefully**: Override `_handle_handler_error` for custom error handling
5. **Use type hints**: Leverage the generic type system for better IDE support

## Migration from Browser-Specific Code

If you're migrating from the browser-specific `BaseWatchdog`:

1. Replace `BrowserSession` with your event processor type
2. Update method names (`attach_to_session` → `attach_to_processor`)
3. Implement the `EventProcessor` protocol on your processor class
4. Update any browser-specific error handling in `_handle_handler_error`

## Example

See `examples/watchdog_example.py` for a complete working example that demonstrates:
- Multiple event types
- Event processor implementation
- Multiple watchdogs working together
- Security monitoring use case
- Comprehensive logging
