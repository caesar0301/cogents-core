"""Example demonstrating the generic watchdog framework."""

import asyncio
import logging

from bubus.models import T_EventResultType
from pydantic import Field

from cogents_core.msgbus import BaseEvent, BaseWatchdog, EventBus


# Example event classes
class UserLoginEvent(BaseEvent[T_EventResultType]):
    """Event emitted when a user logs in."""

    user_id: str = Field()
    timestamp: float = Field()


class UserLogoutEvent(BaseEvent[T_EventResultType]):
    """Event emitted when a user logs out."""

    user_id: str = Field()
    timestamp: float = Field()


class SecurityAlertEvent(BaseEvent[T_EventResultType]):
    """Event emitted when a security alert is triggered."""

    alert_type: str = Field()
    user_id: str = Field()
    severity: str = Field()


# Example event processor (could be a database session, API client, etc.)
class DatabaseSession:
    """Example event processor that manages database operations."""

    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus  # Use the provided event_bus, don't create our own!
        self.logger = logging.getLogger(f"DatabaseSession.{name}")
        self.active_users = set()

    async def login_user(self, user_id: str):
        """Simulate user login."""
        self.active_users.add(user_id)
        self.logger.info(f"User {user_id} logged in")

        # Emit login event
        event = UserLoginEvent(user_id=user_id, timestamp=asyncio.get_event_loop().time())
        self.event_bus.dispatch(event)

    async def logout_user(self, user_id: str):
        """Simulate user logout."""
        if user_id in self.active_users:
            self.active_users.remove(user_id)
            self.logger.info(f"User {user_id} logged out")

            # Emit logout event
            event = UserLogoutEvent(user_id=user_id, timestamp=asyncio.get_event_loop().time())
            self.event_bus.dispatch(event)


# Example watchdog that monitors user activity
class UserActivityWatchdog(BaseWatchdog[DatabaseSession]):
    """Watchdog that monitors user activity and emits security alerts."""

    # Declare which events this watchdog listens to
    LISTENS_TO = [UserLoginEvent, UserLogoutEvent]

    # Declare which events this watchdog emits
    EMITS = [SecurityAlertEvent]

    def __init__(self, event_bus: EventBus, event_processor: DatabaseSession):
        super().__init__(event_bus=event_bus, event_processor=event_processor)
        self._login_count = 0
        self._logout_count = 0

    async def on_UserLoginEvent(self, event: UserLoginEvent):
        """Handle user login events."""
        self._login_count += 1
        self.logger.info(f"User {event.user_id} logged in (total logins: {self._login_count})")

        # Check for suspicious activity (multiple logins in short time)
        if self._login_count > 5:
            alert = SecurityAlertEvent(alert_type="multiple_logins", user_id=event.user_id, severity="medium")
            self.emit_event(alert)
            self.logger.warning(f"Security alert: Multiple logins detected for user {event.user_id}")

    async def on_UserLogoutEvent(self, event: UserLogoutEvent):
        """Handle user logout events."""
        self._logout_count += 1
        self.logger.info(f"User {event.user_id} logged out (total logins: {self._logout_count})")

        # Check for unusual logout patterns
        if self._logout_count > 10:
            alert = SecurityAlertEvent(alert_type="frequent_logouts", user_id=event.user_id, severity="low")
            self.emit_event(alert)
            self.logger.warning(f"Security alert: Frequent logouts detected for user {event.user_id}")


# Example watchdog that logs all events
class EventLoggerWatchdog(BaseWatchdog[DatabaseSession]):
    """Watchdog that logs all events for debugging purposes."""

    # Declare which events this watchdog listens to
    LISTENS_TO = [UserLoginEvent, UserLogoutEvent, SecurityAlertEvent]

    # Don't emit any events
    EMITS = []

    def __init__(self, event_bus: EventBus, event_processor: DatabaseSession):
        super().__init__(event_bus=event_bus, event_processor=event_processor)

    async def on_UserLoginEvent(self, event: UserLoginEvent):
        """Log user login events."""
        self.logger.info(f"🔐 LOGIN: User {event.user_id} at {event.timestamp}")

    async def on_UserLogoutEvent(self, event: UserLogoutEvent):
        """Log user logout events."""
        self.logger.info(f"🚪 LOGOUT: User {event.user_id} at {event.timestamp}")

    async def on_SecurityAlertEvent(self, event: SecurityAlertEvent):
        """Log security alert events."""
        self.logger.warning(f"🚨 ALERT: {event.alert_type} for user {event.user_id} - {event.severity}")


async def main():
    """Main example function."""
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create event bus
        event_bus = EventBus()

        # Create database session with shared event bus
        db_session = DatabaseSession("main_db", event_bus)

        # Create watchdogs
        user_watchdog = UserActivityWatchdog(event_bus=event_bus, event_processor=db_session)
        logger_watchdog = EventLoggerWatchdog(event_bus=event_bus, event_processor=db_session)

        # Attach watchdogs to the processor
        user_watchdog.attach_to_processor()
        logger_watchdog.attach_to_processor()

        # Simulate user activity
        print("🚀 Starting user activity simulation...")

        # Simulate multiple logins
        for i in range(7):
            await db_session.login_user(f"user_{i}")
            await asyncio.sleep(0.1)  # Small delay

        # Simulate some logouts
        for i in range(3):
            await db_session.logout_user(f"user_{i}")
            await asyncio.sleep(0.1)

        # Simulate more logins to trigger security alerts
        for i in range(5):
            await db_session.login_user(f"user_{i+10}")
            await asyncio.sleep(0.1)

        print("✅ User activity simulation completed!")

        # Graceful shutdown sequence to avoid task exception warnings
        print("⏳ Brief wait for events to settle...")
        await asyncio.sleep(0.5)  # Give events time to complete

        print("🛑 Stopping event bus...")

        # First, try graceful stop with short timeout
        try:
            await asyncio.wait_for(event_bus.stop(timeout=1.0, clear=True), timeout=2.0)
            print("✅ Event bus stopped gracefully!")
        except asyncio.TimeoutError:
            print("⚠️  Graceful stop timed out, cleaning up tasks...")

            # Cancel any remaining event bus related tasks
            current_task = asyncio.current_task()
            remaining_tasks = [task for task in asyncio.all_tasks() if not task.done() and task != current_task]

            if remaining_tasks:
                print(f"⏳ Cancelling {len(remaining_tasks)} remaining tasks...")
                for task in remaining_tasks:
                    task.cancel()

                # Wait briefly for cancellation
                try:
                    await asyncio.wait_for(asyncio.gather(*remaining_tasks, return_exceptions=True), timeout=1.0)
                except asyncio.TimeoutError:
                    pass  # Some tasks may still be running, but we tried

            # Now force shutdown
            if hasattr(event_bus, "_is_running"):
                event_bus._is_running = False
            if hasattr(event_bus, "event_queue") and event_bus.event_queue:
                try:
                    event_bus.event_queue.shutdown(immediate=True)
                except Exception:
                    pass

            print("✅ Event bus stopped!")
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")
            print("✅ Event bus stopped with errors!")

    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("🏁 Main function completed")


if __name__ == "__main__":
    asyncio.run(main())
