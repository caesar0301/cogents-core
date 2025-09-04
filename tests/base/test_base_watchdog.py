"""Unit tests for BaseWatchdog - simplified version that avoids hangs."""

import logging
from unittest.mock import MagicMock, patch

import pytest
from bubus import BaseEvent, EventBus
from bubus.models import T_EventResultType
from pydantic import Field, PrivateAttr, ValidationError

from cogents.core.base.msgbus.base import BaseWatchdog


# Test Events
class MockTestEvent(BaseEvent[T_EventResultType]):
    """Test event for unit tests."""

    data: str = Field()


class AnotherMockTestEvent(BaseEvent[T_EventResultType]):
    """Another test event."""

    value: int = Field()


class SecurityAlertEvent(BaseEvent[T_EventResultType]):
    """Security alert event for emission tests."""

    alert_type: str = Field()
    severity: str = Field()


# Test Event Processors
class MockEventProcessor:
    """Mock event processor that implements EventProcessor protocol."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.logger = logging.getLogger("MockEventProcessor")


class InvalidEventProcessor:
    """Event processor without event_bus attribute."""

    def __init__(self):
        self.logger = logging.getLogger("InvalidEventProcessor")


class SeparateBusEventProcessor:
    """Event processor with separate event bus instance."""

    def __init__(self):
        self.event_bus = EventBus()  # Creates separate instance
        self.logger = logging.getLogger("SeparateBusEventProcessor")


# Test Watchdogs
class MockTestWatchdog(BaseWatchdog[MockEventProcessor]):
    """Test watchdog for unit tests."""

    LISTENS_TO = [MockTestEvent, AnotherMockTestEvent]
    EMITS = [SecurityAlertEvent]

    # Use PrivateAttr for test state
    _handled_events: list = PrivateAttr(default_factory=list)
    _handler_call_count: int = PrivateAttr(default=0)

    @property
    def handled_events(self) -> list:
        return self._handled_events

    @property
    def handler_call_count(self) -> int:
        return self._handler_call_count

    async def on_MockTestEvent(self, event: MockTestEvent) -> str:
        """Handle test events."""
        self._handled_events.append(event)
        self._handler_call_count += 1
        return f"Handled: {event.data}"

    async def on_AnotherMockTestEvent(self, event: AnotherMockTestEvent) -> str:
        """Handle another test events."""
        self._handled_events.append(event)
        self._handler_call_count += 1
        return f"Handled: {event.value}"


class EmptyWatchdog(BaseWatchdog[MockEventProcessor]):
    """Watchdog with no handlers."""

    LISTENS_TO = []
    EMITS = []


class MissingHandlerWatchdog(BaseWatchdog[MockEventProcessor]):
    """Watchdog with LISTENS_TO but missing handler methods."""

    LISTENS_TO = [MockTestEvent]
    EMITS = []

    # No on_MockTestEvent method defined


# Fixtures
@pytest.fixture
def event_bus():
    """Provide a fresh EventBus instance for each test."""
    return EventBus()


@pytest.fixture
def mock_processor(event_bus):
    """Provide a mock processor with the same event bus."""
    return MockEventProcessor(event_bus)


class TestBaseWatchdogValidation:
    """Test BaseWatchdog validation functionality."""

    def test_valid_initialization(self, event_bus, mock_processor):
        """Test that valid BaseWatchdog initialization works."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        assert watchdog.event_bus is event_bus
        assert watchdog.event_processor is mock_processor
        assert watchdog.LISTENS_TO == [MockTestEvent, AnotherMockTestEvent]
        assert watchdog.EMITS == [SecurityAlertEvent]

    def test_missing_event_bus_attribute(self, event_bus):
        """Test validation fails when event_processor lacks event_bus attribute."""
        processor = InvalidEventProcessor()

        with pytest.raises(ValidationError) as exc_info:
            MockTestWatchdog(event_bus=event_bus, event_processor=processor)

        error_msg = str(exc_info.value)
        assert "must have an 'event_bus' attribute" in error_msg
        assert "EventProcessor protocol" in error_msg

    def test_different_event_bus_instances(self, event_bus):
        """Test validation fails when event_processor has different event_bus."""
        processor = SeparateBusEventProcessor()

        with pytest.raises(ValidationError) as exc_info:
            MockTestWatchdog(event_bus=event_bus, event_processor=processor)

        error_msg = str(exc_info.value)
        assert "must be the same instance" in error_msg
        assert "infinite hangs" in error_msg

    def test_same_event_bus_instance_accepted(self):
        """Test that same event_bus instance is accepted."""
        shared_bus = EventBus()
        processor = MockEventProcessor(shared_bus)

        # Should not raise any exception
        watchdog = MockTestWatchdog(event_bus=shared_bus, event_processor=processor)
        assert watchdog.event_bus is shared_bus
        assert watchdog.event_processor.event_bus is shared_bus


class TestBaseWatchdogAttachment:
    """Test BaseWatchdog attachment and handler registration."""

    def test_attach_to_processor_success(self, event_bus, mock_processor):
        """Test successful attachment with handler registration."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        with patch.object(watchdog.logger, "info") as mock_log:
            watchdog.attach_to_processor()

            # Should log successful registration
            mock_log.assert_called_with("[MockTestWatchdog] Successfully registered 2 event handlers")

    def test_empty_listens_to_warning(self, event_bus, mock_processor):
        """Test warning when LISTENS_TO is empty."""
        watchdog = EmptyWatchdog(event_bus=event_bus, event_processor=mock_processor)

        with patch.object(watchdog.logger, "warning") as mock_warn:
            watchdog.attach_to_processor()

            # Should log warning about no LISTENS_TO
            mock_warn.assert_called()
            warning_msg = mock_warn.call_args[0][0]
            assert "No event classes discovered" in warning_msg
            assert "Define LISTENS_TO" in warning_msg

    def test_missing_handler_warning(self, event_bus, mock_processor):
        """Test warning when handler method is missing for declared event."""
        watchdog = MissingHandlerWatchdog(event_bus=event_bus, event_processor=mock_processor)

        with patch.object(watchdog.logger, "warning") as mock_warn:
            watchdog.attach_to_processor()

            # Should warn about missing handler
            mock_warn.assert_called()
            warning_calls = [call[0][0] for call in mock_warn.call_args_list]
            assert any("but no handlers found" in msg for msg in warning_calls)

    def test_event_processor_not_initialized(self, event_bus, mock_processor):
        """Test assertion when event_processor is None."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)
        watchdog.event_processor = None

        with pytest.raises(AssertionError, match="Event processor not initialized"):
            watchdog.attach_to_processor()


class TestBaseWatchdogHandlerRegistration:
    """Test BaseWatchdog handler registration without async execution."""

    def test_handler_registration_validation(self, event_bus, mock_processor):
        """Test that handlers are properly registered during attachment."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        # Before attachment, no handlers should be registered
        assert MockTestEvent.__name__ not in event_bus.handlers
        assert AnotherMockTestEvent.__name__ not in event_bus.handlers

        # Attach handlers
        watchdog.attach_to_processor()

        # After attachment, handlers should be registered
        assert MockTestEvent.__name__ in event_bus.handlers
        assert AnotherMockTestEvent.__name__ in event_bus.handlers
        assert len(event_bus.handlers[MockTestEvent.__name__]) == 1
        assert len(event_bus.handlers[AnotherMockTestEvent.__name__]) == 1

    def test_handler_method_detection(self, event_bus, mock_processor):
        """Test that handler methods are correctly detected."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        # Check that handler methods exist
        assert hasattr(watchdog, "on_MockTestEvent")
        assert hasattr(watchdog, "on_AnotherMockTestEvent")
        assert callable(watchdog.on_MockTestEvent)
        assert callable(watchdog.on_AnotherMockTestEvent)

    def test_handler_wrapper_creation(self, event_bus, mock_processor):
        """Test that handler wrappers are created with unique names."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)
        watchdog.attach_to_processor()

        # Get the registered handlers
        mock_handlers = event_bus.handlers[MockTestEvent.__name__]
        another_handlers = event_bus.handlers[AnotherMockTestEvent.__name__]

        # Check that handlers have unique names
        assert mock_handlers[0].__name__ == "MockTestWatchdog.on_MockTestEvent"
        assert another_handlers[0].__name__ == "MockTestWatchdog.on_AnotherMockTestEvent"


class TestBaseWatchdogEventEmission:
    """Test BaseWatchdog event emission functionality."""

    def test_emit_event_success(self, event_bus, mock_processor):
        """Test successful event emission."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        # Mock the dispatch method
        with patch.object(event_bus, "dispatch") as mock_dispatch:
            alert_event = SecurityAlertEvent(alert_type="test", severity="low")
            watchdog.emit_event(alert_event)

            mock_dispatch.assert_called_once_with(alert_event)

    def test_emit_undeclared_event_error(self, event_bus, mock_processor):
        """Test error when emitting undeclared event."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        # Try to emit event not in EMITS
        undeclared_event = MockTestEvent(data="undeclared")

        with pytest.raises(AssertionError) as exc_info:
            watchdog.emit_event(undeclared_event)

        error_msg = str(exc_info.value)
        assert "not declared in EMITS" in error_msg
        assert "MockTestEvent" in error_msg

    def test_emit_event_no_emits_validation(self, event_bus, mock_processor):
        """Test emit_event works when EMITS is empty (no validation)."""
        watchdog = EmptyWatchdog(event_bus=event_bus, event_processor=mock_processor)

        # Should work without validation when EMITS is empty
        with patch.object(event_bus, "dispatch") as mock_dispatch:
            test_event = MockTestEvent(data="no_validation")
            watchdog.emit_event(test_event)

            mock_dispatch.assert_called_once_with(test_event)


class TestBaseWatchdogStaticMethods:
    """Test BaseWatchdog static methods."""

    def test_attach_handler_to_processor_success(self, event_bus, mock_processor):
        """Test successful handler attachment."""

        # Mock handler method
        async def mock_handler(event: MockTestEvent) -> str:
            return f"Mock handled: {event.data}"

        mock_handler.__name__ = "on_MockTestEvent"
        mock_handler.__self__ = MagicMock()
        mock_handler.__self__.__class__.__name__ = "MockTestWatchdog"

        # Test attachment
        BaseWatchdog.attach_handler_to_processor(mock_processor, MockTestEvent, mock_handler)

        # Verify handler is registered
        assert MockTestEvent.__name__ in event_bus.handlers
        assert len(event_bus.handlers[MockTestEvent.__name__]) == 1

    def test_attach_handler_invalid_name(self, event_bus, mock_processor):
        """Test handler attachment fails with invalid handler name."""

        async def bad_handler(event: MockTestEvent) -> str:
            return "bad"

        bad_handler.__name__ = "bad_name"  # Doesn't start with "on_"

        with pytest.raises(AssertionError, match='must start with "on_"'):
            BaseWatchdog.attach_handler_to_processor(mock_processor, MockTestEvent, bad_handler)

    def test_attach_handler_wrong_event_type_suffix(self, event_bus, mock_processor):
        """Test handler attachment fails when name doesn't end with event type."""

        async def wrong_handler(event: MockTestEvent) -> str:
            return "wrong"

        wrong_handler.__name__ = "on_WrongEvent"  # Doesn't end with "MockTestEvent"

        with pytest.raises(AssertionError, match="must end with event type"):
            BaseWatchdog.attach_handler_to_processor(mock_processor, MockTestEvent, wrong_handler)

    def test_duplicate_handler_registration(self, event_bus, mock_processor):
        """Test error on duplicate handler registration."""

        async def handler1(event: MockTestEvent) -> str:
            return "handler1"

        async def handler2(event: MockTestEvent) -> str:
            return "handler2"

        # Set up handlers with same name (simulating same watchdog class)
        for handler in [handler1, handler2]:
            handler.__name__ = "on_MockTestEvent"
            handler.__self__ = MagicMock()
            handler.__self__.__class__.__name__ = "MockTestWatchdog"

        # First registration should succeed
        BaseWatchdog.attach_handler_to_processor(mock_processor, MockTestEvent, handler1)

        # Second registration should fail
        with pytest.raises(RuntimeError, match="Duplicate handler registration"):
            BaseWatchdog.attach_handler_to_processor(mock_processor, MockTestEvent, handler2)


class TestBaseWatchdogProperties:
    """Test BaseWatchdog properties and methods."""

    def test_logger_property(self, event_bus, mock_processor):
        """Test that logger property returns event_processor logger."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        assert watchdog.logger is mock_processor.logger

    def test_del_method_task_cleanup(self, event_bus, mock_processor):
        """Test __del__ method cleans up asyncio tasks."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        # Mock asyncio task
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        mock_task.done.return_value = False

        # Add mock task to watchdog
        watchdog._test_task = mock_task

        # Call __del__
        watchdog.__del__()

        # Verify task was cancelled
        mock_task.cancel.assert_called_once()

    def test_del_method_task_collection_cleanup(self, event_bus, mock_processor):
        """Test __del__ method cleans up task collections."""
        watchdog = MockTestWatchdog(event_bus=event_bus, event_processor=mock_processor)

        # Mock task collection
        mock_tasks = []
        for i in range(3):
            task = MagicMock()
            task.cancel = MagicMock()
            task.done.return_value = False
            mock_tasks.append(task)

        # Add mock task collection to watchdog
        watchdog._test_tasks = mock_tasks

        # Call __del__
        watchdog.__del__()

        # Verify all tasks were cancelled
        for task in mock_tasks:
            task.cancel.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
