"""Components, identification, counters and locking of a device."""

from __future__ import annotations

import asyncio
from asyncua import Node
from typing import Self, TYPE_CHECKING
from .ids import SubscriptionLevel
from .log import _logger
from .node import LADSNode, promote_to
from .sets import ComponentSet
from .util import remove_none
from .variables import Enumeration, LifetimeCounter, get_properties_and_variables

if TYPE_CHECKING:
    from .server import Server
    from .sets import LADSSet
    from .variables import BaseVariable


# MARK: OperationCounters
class OperationCounters(LADSNode):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(OperationCounters, node, server.MachineryOperationCounterType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.operation_cycle_counter, self.operation_duration, self.power_on_duration = await asyncio.gather(
            self.get_di_variable("OperationCycleCounter"),
            self.get_di_variable("OperationDuration"),
            self.get_di_variable("PowerOnDuration"),
        )
        for variable in self.variables:
            variable.subscription_level = SubscriptionLevel.Temporary

    @property
    def variables(self) -> list[BaseVariable]:
        return remove_none([self.operation_cycle_counter, self.operation_duration, self.power_on_duration])


# MARK: LifetimeCounters
class LifetimeCounters(LADSNode):
    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(LifetimeCounters, node, server.MachineryLifeTimeCounterType, server)

    async def init(self, server: Server):
        await super().init(server)
        nodes = await get_properties_and_variables(self)
        self.lifetime_counters: list[LifetimeCounter] = await asyncio.gather(*(LifetimeCounter.promote(node, server) for node in nodes))
        self.lifetime_counters.sort(key = lambda node: node.display_name)

    @property
    def variables(self) -> list[Node]:
        return super().variables + self.lifetime_counters


# MARK: Identification
class Identification(LADSNode):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Identification, node, server.MachineryItemIdentificationType, server)

    async def init(self, server: Server):
        self.asset_id: BaseVariable = None
        self.component_name: BaseVariable = None
        self.location: BaseVariable = None

        await super().init(server)
        self._variables = await get_properties_and_variables(self)
        self._variables.sort(key = lambda variable: variable.display_name)
        self.asset_id = self.variable_named("AssetId")
        self.component_name = self.variable_named("ComponentName")
        self.location = self.variable_named("Location")
        subscription_variables: list[BaseVariable] = remove_none([self.asset_id, self.component_name, self.location])
        for variable in subscription_variables:
            variable.subscription_level = SubscriptionLevel.Permanent

    @property
    def variables(self) ->list[BaseVariable]:
        return self._variables


# MARK: Component
class Component(LADSNode):

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Component, node, server.ComponentType, server)

    async def init(self, server: Server):
        self.component_set: LADSSet = None
        self.device_health: Enumeration = None
        self.operation_counters: OperationCounters = None
        self.lifetime_counter_set: LifetimeCounters = None
        self.identification: Identification = None
        await super().init(server)

        self.operation_counters = await OperationCounters.promote(await self.get_di_child("OperationCounters"), server)
        self.lifetime_counter_set = await LifetimeCounters.promote(await self.get_machinery_child("LifetimeCounters"), server)
        self.identification = await Identification.promote(await self.get_di_child("Identification"), server)
        self._variables = await get_properties_and_variables(self)
        if self.operation_counters is not None:
            self._variables = self._variables + self.operation_counters.variables
        self._variables.sort(key = lambda variable: variable.display_name)
        self.device_health = self.variable_named("DeviceHealth")
        if self.device_health is not None:
            self.device_health = await Enumeration.promote(self.device_health, server)
        self.component_set = await ComponentSet.promote(await self.get_machinery_child("Components"), server)
        if self.component_set is not None:
            await self.component_set.promote_children(Component, server.ComponentType, server.ComponentSetType)

    @property
    def components(self) -> list[Component]:
        return [] if self.component_set is None else self.component_set.children

    @property
    def lifetime_counters(self) -> list[LifetimeCounter]:
        return [] if self.lifetime_counter_set is None else self.lifetime_counter_set.lifetime_counters

    @property
    def variables(self) ->list[BaseVariable]:
        ltc_variables: list[BaseVariable] = []
        for ltc in self.lifetime_counters:
            ltc_variables.append(ltc)
        return self._variables + ltc_variables + [] if self.component_set is None else remove_none([self.component_set.node_version])

    @property
    def name_plate_variables(self) ->list[BaseVariable]:
        return self._variables if self.identification is None else remove_none([self.device_health] + self.identification.variables)


# MARK: Lock
class Lock(LADSNode):
    INIT_LOCK = "InitLock"
    EXIT_LOCK = "ExitLock"
    BREAK_LOCK = "BreakLock"
    RENEW_LOCK = "RenewLock"
    COMMANDS = [INIT_LOCK, EXIT_LOCK, BREAK_LOCK, RENEW_LOCK]

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Lock, node, server.LockingServicesType, server)

    async def init(self, server: Server):
        await super().init(server)
        self.max_inactive_lock_time = server.max_inactive_lock_time
        (self.locked, self.locking_client, self.locking_user, self.remaining_lock_time) = await asyncio.gather(
            self.get_di_variable("Locked"),
            self.get_di_variable("LockingClient"),
            self.get_di_variable("LockingUser"),
            self.get_di_variable("RemainingLockTime"),
        )
        parent = await LADSNode.promote(await self.get_parent(), server)
        _logger.debug(f"Locking service of {parent.display_name} initialized")

    def init_lock(self, context:str = "LADS Cient"):
        self.call_async(self.call_di_method(Lock.INIT_LOCK, context))

    def exit_lock(self):
        self.call_async(self.call_di_method(Lock.EXIT_LOCK))

    def break_lock(self):
        self.call_async(self.call_di_method(Lock.BREAK_LOCK))

    def renew_lock(self):
        self.call_async(self.call_di_method(Lock.RENEW_LOCK))

    def call_method_by_name(self, name: str, *args):
        if name is None:
            return
        match name:
            case Lock.INIT_LOCK:
                self.init_lock()
            case Lock.EXIT_LOCK:
                self.exit_lock()
            case Lock.BREAK_LOCK:
                self.break_lock()
            case Lock.RENEW_LOCK:
                self.renew_lock()

    @property
    def variables(self) ->list[BaseVariable]:
        return remove_none([self.locked, self.locking_client, self.locking_user, self.remaining_lock_time])
