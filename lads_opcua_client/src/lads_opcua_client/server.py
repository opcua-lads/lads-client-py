"""Server: a LADS OPC UA server with its devices and functional units."""

from __future__ import annotations

import asyncio
from asyncua import Client, ua
from queue import Queue
from typing import TYPE_CHECKING
from .device import Device
from .ids import DIObjectIds
from .lads_types import LADSTypes
from .log import _logger
from .variables import BaseVariable

if TYPE_CHECKING:
    from .functional_unit import FunctionalUnit


# MARK: Server
class Server(LADSTypes):
    """
    Represents a LADS OPC UA server with a set of devices and functional units.

    Attributes:
        client: Client - the client object used to connect to the server
        name: str - the name of the server
        devices: list[Device] - the devices in the server
        initialized: bool - True if the server has been initialized
        running: bool - True if the server is running
        call_async_queue: Queue - queue for async calls
        functional_units: list[FunctionalUnit] - all functional units in the server
    """

    def __init__(self, client: Client):
        super().__init__(client)
        self.name = "Unknown Server"
        self.devices: list[Device] = []
        self.initialized = False
        self.running = True
        self.call_async_queue = Queue()

    async def _disconnect(self):
        print(f"Disconnecting from {self.name}")
        await self.client.disconnect()
        self.running = False
        self.initialized = False
        print("Disconnected successfully.")

    async def init(self) -> dict:
        data_types = await super().init()

        # browse for devices in DeviceSet
        try:
            product_uri = self.client.get_node(ua.ObjectIds.Server_ServerStatus_BuildInfo_ProductUri)
            self.name: str = await product_uri.read_value()
            _logger.info(f"Server product URI {self.name}")
        except Exception as error:
            _logger.error("Unable to read product URI")

        # locking services
        try:
            self.max_inactive_lock_time = await BaseVariable.promote(self.get_di_node(DIObjectIds.MaxInactiveLockTime), self)
            if self.max_inactive_lock_time is not None:
                await self.max_inactive_lock_time.read_data_value()
        except Exception as error:
            _logger.warn("Unable to read inactive lock time")

        # devices
        device_set = await self.client.nodes.objects.get_child(f"{self.ns_DI}:DeviceSet")
        nodes = await device_set.get_children(refs = ua.ObjectIds.HasChild, nodeclassmask = ua.NodeClass.Object)
        for node in nodes:
            try:
                await self.client.check_connection()
                device: Device = await Device.promote(node, self)
                await device.finalize_init()
                self.devices.append(device)
            except Exception as error:
                _logger.error(error, extra=["node", node])
                return data_types

        self.initialized = True


        return data_types

    async def evaluate(self):
        if not self.call_async_queue.empty():
            item = self.call_async_queue.get()
            if item is not None:
                try:
                    result = await item
                    _logger.debug(f"Evaluated item {item} with result {result}")
                except Exception as error:
                    _logger.debug(error)
        await asyncio.sleep(0.01)

    @property
    def functional_units(self) -> list[FunctionalUnit]:
        if not self.initialized: return []
        functional_units: list[FunctionalUnit] = []
        for device in self.devices:
            functional_units = functional_units + device.functional_units
        return functional_units
