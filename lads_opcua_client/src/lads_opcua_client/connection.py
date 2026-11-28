"""Connection and Connections: connecting to one or more LADS OPC UA servers."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from asyncua import Client, ua
from typing import TYPE_CHECKING
from .config import expand_env, get_value
from .log import _logger
from .server import Server

if TYPE_CHECKING:
    from .functional_unit import FunctionalUnit


RECONNECT_SECONDS = 2


# MARK: Connection
class Connection:
    """
    Connection class for managing a LADS OPC UA client-server connection.

    Attributes:
        client (Client): The LADS OPC UA client instance.
        server (Server): The LADS OPC UA server instance.
        url (str): The URL of the LADS OPC UA server.
        user (str): The username for authentication.
        password (str): The password for authentication.
        thread (threading.Thread): The thread responsible for running the connection.

    Methods:
        initialized:
            Checks if the server is initialized.
        connect():
            Starts the connection thread (non-asynchronous) and waits until the server is initialized.
        disconnect():
            Disconnects the server (non-asynchronous).
    """

    data_types: dict = None

    def __init__(self, url = None, user: str = None, password: str = None) -> None:
        """
        Prepares the connection to a LADS OPC UA server.

        Args:
            url (str): The URL of the LADS OPC UA server. Defaults to None.
            user (str, optional): The username for authentication. Defaults to None.
            password (str, optional): The password for authentication. Defaults to None.
        """

        self.client: Client = None
        self.server: Server = None
        self.url = url
        self.user = user
        self.password = password
        self.running = False
        self.last_error: str | None = None
        self.thread = threading.Thread(target=self._run_connection, daemon=True, name=f"LADS OPC UA Connection {url}")
        #self.thread.start()

    @property
    def initialized(self) -> bool:
        """Checks if the server is initialized."""
        if self.server is None:
            return False
        else:
            return self.server.initialized

    @property
    def settled(self) -> bool:
        """True once the first connection attempt has either succeeded or failed."""
        return self.initialized or self.last_error is not None

    def _run_connection(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_connection_async())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def connect(self):
        """Starts the connection thread (non-asynchronous)."""
        if self.thread.is_alive():
            return

        if not self.thread.is_alive() and self.thread._started.is_set():
            self.thread = threading.Thread(
                target=self._run_connection, daemon=True, name=f"LADS OPC UA Connection {self.url}")

        self.running = True
        self.thread.start()

    def stop(self):
        """Requests the connection thread to stop without waiting for it."""
        self.running = False

    def disconnect(self):
        """Disconnects the server and stops the thread (non-asynchronous)."""
        self.stop()
        if self.thread.is_alive():
            print(f"Disconnecting from {self.url}...", end="")
            self.thread.join(timeout=10)
            print("Done!")

    async def _sleep_while_running(self, seconds: float):
        """Sleeps, but wakes up early when the connection is asked to stop."""
        end = time.monotonic() + seconds
        while self.running and time.monotonic() < end:
            await asyncio.sleep(0.1)

    def _report_error(self, error: Exception, unexpected: bool = False):
        """Logs a connection problem once per distinct message instead of on every retry."""
        message = str(error) or type(error).__name__
        if message != self.last_error:
            _logger.warning(f"{self.url}: {message} - retrying every {RECONNECT_SECONDS} s",
                            exc_info=error if unexpected else None)
        self.last_error = message

    async def _run_connection_async(self):
        while self.running:
            self.client = Client(self.url)
            self.server = Server(self.client)
            if (self.user is not None) and (self.password is not None):
                self.client.set_user(self.user)
                self.client.set_password(self.password)
            try:
                async with self.client:
                    await self.server.init()
                    self.last_error = None
                    while self.server.running and self.running:
                        await self.server.evaluate()
                        await self.client.check_connection()
                    if self.server.running:
                        self.server.running = False
                        self.server.initialized = False
            except (OSError, ua.UaError) as error:
                # OSError covers TimeoutError, ConnectionError and asyncio's "Multiple exceptions" (IPv4/IPv6)
                self._report_error(error)
                await self._sleep_while_running(RECONNECT_SECONDS)
            except Exception as error:
                self._report_error(error, unexpected=True)
                await self._sleep_while_running(RECONNECT_SECONDS)


# MARK: Connections
class Connections:
    """
    Connections class for managing multiple LADS OPC UA client-server connections.

    Attributes:
        connections (list[Connection]): The list of connections.
        urls (list[str]): The list of URLs of the connections.
        initialized (bool): Checks if all servers are initialized.
        functional_units (list[FunctionalUnit]): The list of functional units from all connections.

    Methods:
        add(url, user, password):
            Adds a new connection with the given parameters.
        connect():
            Starts all connection threads (non-asynchronous).
        disconnect():
            Disconnects from all server connections (non-asynchronous).
    """

    def __init__(self, config_file = "config.json") -> None:
        """
        Initializes the connections with the given configuration file.

        Args:
            config_file (str): The path to the configuration file. Defaults to "config.json".
        """

        self.connections: list[Connection] = []
        try:
            with open(config_file) as f:
                print(f"parsing config file {config_file}")
                data = json.load(f)
        except Exception as error:
            _logger.error(f"Invalid config file {config_file}: {error}")
            return
        for connection in data.get("connections", []):
            enabled = get_value(connection, "enabled")
            if enabled is None:
                enabled = True
            if not enabled:
                continue
            try:
                url = expand_env(connection["url"])
                user = expand_env(get_value(connection, "user"))
                password = expand_env(get_value(connection, "password"))
            except (KeyError, TypeError) as error:
                # skip only the affected connection, e.g. when an environment variable is not set
                _logger.error(f"Skipping connection {get_value(connection, 'url')}: {error}")
                continue
            print(f"Add conncetion with url {url}")
            self.add(url, user, password)

    def add(self, url: str, user: str = None, password: str = None) -> Connection:
        """
        Adds a new connection with the given parameters.

        Args:
            url (str): The URL of the OPC UA server.
            user (str, optional): The username for authentication. Defaults to None.
            password (str, optional): The password for authentication. Defaults to None.
        """

        connection = Connection(url, user, password)
        self.connections.append(connection)
        return connection

    def connect(self):
        """Starts the connection threads (non-asynchronous)."""
        for connection in self.connections:
            connection.connect()

    def disconnect(self):
        """Disconnects the server/s (non-asynchronous)."""
        for connection in self.connections:
            connection.stop()
        for connection in self.connections:
            connection.disconnect()

    @property
    def urls(self) -> list[str]:
        return list(map(lambda connection: connection.url, self.connections))

    @property
    def settled(self) -> bool:
        """True once every connection has either been initialized or failed its first attempt."""
        return len(self.connections) > 0 and all(connection.settled for connection in self.connections)

    @property
    def any_initialized(self) -> bool:
        return any(connection.initialized for connection in self.connections)

    @property
    def unreachable(self) -> dict[str, str]:
        """URLs of connections that are not initialized (yet), with their last error. They keep retrying."""
        return {connection.url: connection.last_error for connection in self.connections
                if not connection.initialized and connection.last_error is not None}

    @property
    def initialized(self) -> bool:
        if len(self.connections) == 0:
            return False
        result = True
        for connection in self.connections:
            result = result and connection.initialized
        return result

    @property
    def functional_units(self) -> list[FunctionalUnit]:
        result: list[FunctionalUnit] = []
        for connection in self.connections:
            if connection.initialized:
                result.extend(connection.server.functional_units)
        return result
