"""Device: a LADS device with its components, functional units and state."""

from __future__ import annotations

import asyncio
from asyncua import Node
from asyncua.common.events import Event
from typing import Self, TYPE_CHECKING, Tuple
from .compliance import ComplianceDocument, ComplianceDocumentSet
from .component import Component, Lock
from .functional_unit import FunctionalUnit
from .ids import SubscriptionLevel
from .log import _logger
from .node import promote_to
from .state_machine import StateMachine
from .subscription import SubscriptionHandler
from .util import remove_none, unique_name_delimiter
from .variables import BaseVariable

if TYPE_CHECKING:
    from .server import Server
    from .sets import LADSSet
    from .variables import SubscribedVariable


# MARK: Device
class Device(Component):
    """
    Represents a device in the LADS OPC UA model.

    Attributes:
        device_state: StateMachine - State machine representing the device state.
        machinery_item_state: StateMachine - State machine representing the machinery item state.
        machinery_operation_mode: StateMachine - State machine representing the machinery operation mode.
        location: SubscribedVariable - Location of the device.
        hierarchical_location: SubscribedVariable - Hierarchical location of the device.
        operational_location: SubscribedVariable - Operational location of the device.
        state_machine_variables: list[BaseVariable] - List of state machine variables of the device.
        location_variables: list[BaseVariable] - List of location variables of the device.
        geographical_location: Tuple[float, float] - Geographical location of the device.
        unique_name: str - Unique name of the device.
        variables: list[BaseVariable] - List of variables of the device.
        events: list[Event] - List of events of the device.
    """

    @classmethod
    async def promote(cls, node: Node, server: Server) -> Self:
        return await promote_to(Device, node, server.DeviceType, server)

    async def init(self, server: Server):
        self.device_state: StateMachine = None
        self.machinery_item_state: StateMachine = None
        self.machinery_operation_mode: StateMachine = None
        self.location: SubscribedVariable = None
        self.hierarchical_location: SubscribedVariable = None
        self.operational_location: SubscribedVariable = None
        self.state_machine_variables: list[BaseVariable] = []
        self.device_type_images = []
        self.compliance_document_set: LADSSet = None
        await super().init(server)

        functional_unit_set = await self.get_lads_child("FunctionalUnitSet")
        nodes = await self.get_child_objects(functional_unit_set)
        self.functional_units: list[FunctionalUnit] = await asyncio.gather(*(FunctionalUnit.promote(node, server) for node in nodes))
        self.device_state, self.machinery_item_state, self.machinery_operation_mode, self.lock = await asyncio.gather(
            StateMachine.promote(await self.get_lads_child("DeviceState"), server),
            StateMachine.promote(await self.get_machinery_child("MachineryItemState"), server),
            StateMachine.promote(await self.get_machinery_child("MachineryOperationMode"), server),
            Lock.promote(await self.get_di_child("Lock"), server)
        )
        state_machines: list[StateMachine] = remove_none([self.device_state, self.machinery_item_state, self.machinery_operation_mode])
        self.state_machine_variables = list(map(lambda state_machine: state_machine.current_state, state_machines))
        if self.device_health is not None:
            self.state_machine_variables.append(self.device_health)

        # location
        self.hierarchical_location, self.operational_location = await asyncio.gather(
            self.get_amb_variable("HierarchicalLocation"),
            self.get_amb_variable("OperationalLocation"),
        )
        self.location = None if self.identification is None else self.identification.location
        for location in self.location_variables:
            location.subscription_level = SubscriptionLevel.Temporary

        # device type images
        self.device_type_images = []
        device_type_image = await self.get_di_child("DeviceTypeImage")
        if device_type_image is not None:
            nodes = await device_type_image.get_variables()
            for node in nodes:
                variable = await BaseVariable.promote(node, server)
                value = variable.value
                from sys import getsizeof
                if value != None and getsizeof(value) > 0:
                    self.device_type_images.append(value)

        # compliance documents
        if server.ns_LADS_CD is not None:
            self.compliance_document_set = await ComplianceDocumentSet.promote(await self.get_lads_cd_child("ComplianceDocumentSet"), self.server)
            if self.compliance_document_set is not None:
                _logger.debug("loading compliance documents")
                await self.compliance_document_set.promote_children(ComplianceDocument, self.server.ComplianceDocumentType, self.server.ComplianceDocumentSetType)
            else:
                _logger.debug("unable to find compliance document set")


    async def finalize_init(self):
        await super().finalize_init()
        await asyncio.gather(*(functional_unit.finalize_init(self) for functional_unit in self.functional_units))
        # prepare subscriptions
        variables = self.subscribed_variables + self.location_variables
        for functional_unit in self.functional_units:
            variables = variables + functional_unit.all_subscribed_variables
        if self.identification is not None:
            variables = variables + self.identification.subscribed_variables
        if self.lock is not None:
            variables = variables + self.lock.variables
        for component in self.components:
            variables = variables + component.subscribed_variables
            if component.identification is not None:
                variables = variables + component.identification.subscribed_variables
            if component.operation_counters is not None:
                variables = variables + component.operation_counters.subscribed_variables
            for lifetime_counter in component.lifetime_counters:
                variables = variables + lifetime_counter.subscribed_variables
        if self.operation_counters is not None:
            variables = variables + self.operation_counters.subscribed_variables
        self.subscription_handler = SubscriptionHandler()
        variable_set = set(variables)
        _logger.debug(f"Device {self.display_name} subscribing to {len(variable_set)}/{len(variables)} variables")
        data_change_handlers = await self.subscription_handler.subscribe_data_change(self.server, variable_set)
        try:
            events_handler = await self.subscription_handler.subscribe_events(self.server, self)
        except:
            try:
                events_handler = await self.subscription_handler.subscribe_events(self.server, self.server.client.get_server_node())
            except:
                _logger.warning("Unable to subscribe to events")


    @property
    def location_variables(self) ->list[BaseVariable]:
        return remove_none([self.location, self.hierarchical_location, self.operational_location])

    @property
    def geographical_location(self) -> Tuple[float, float] | None:
        location = self.location if self.location is not None else self.operational_location
        # location = self.operational_location
        if location is not None:
            try:
                position = location.value_str
                l = position.split(" ")
                if len(l) == 4:
                    lon = float(l[1]) * (-1 if "S" in l[0].upper() else 1)
                    lat = float(l[3]) * (-1 if "W" in l[0].upper() else 1)
                    return (lon, lat)
            except:
                return None
        return None

    @property
    def unique_name(self) -> str:
        return f"{self.server.name}{unique_name_delimiter}{self.display_name}"

    @property
    def variables(self) ->list[BaseVariable]:
        document_set_node_version = self.compliance_document_set.node_version if self.compliance_document_set is not None else None
        _logger.debug(document_set_node_version)
        vars = self.name_plate_variables + self.state_machine_variables
        vars.append(self.compliance_document_set.node_version if self.compliance_document_set is not None else None)
        return remove_none(vars)

    @property
    def events(self) ->list[Event]:
        if self.subscription_handler is not None:
            return self.subscription_handler.event_list
        else:
            return []
