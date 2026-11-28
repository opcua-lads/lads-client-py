"""SubscriptionHandler: data change and event notifications for subscribed variables."""

from __future__ import annotations

import datetime as dt
import pandas as pd
from asyncua import Node
from asyncua.common.events import Event
from asyncua.common.subscription import DataChangeNotif
from typing import Any, TYPE_CHECKING
from .log import _logger
from .util import remove_none, variant_value_to_str

if TYPE_CHECKING:
    from .node import LADSNode
    from .server import Server
    from .variables import BaseVariable


# MARK: SubscriptionHandler
# queue size of the event subscriptions: events raised in a burst (e.g. a state change of a functional unit and its
# controllers) within one publishing interval
EVENT_QUEUE_SIZE = 1000


class SubscriptionHandler(object):
    """
    Handles data change and event notifications for subscribed variables.

    Attributes:
        subscription: Subscription - the subscription object from the asyncua library.
        subscribed_variables: dict[ua.NodeId, BaseVariable] - the subscribed variables as a dictionary.
        event_node: LADSNode - the event LADS node.
        events: pd.DataFrame - the events as a pandas dataframe.
        last_event_update: dt.datetime - the last event update time.

    Methods:
        subscribe_data_change(self, server: Server, nodes: list[BaseVariable], period: float = 500) - subscribe to data change notifications.
        subscribe_events(self, server: Server, node: Node, period: float = 500) - subscribe to event notifications.
        datachange_notification(self, node: Node, val: Any, data: DataChangeNotif) - handle data change notifications.
        event_notification(self, event: Event) - Notification of an event.
        status_change_notification(self, status: Any) - Notification of a status change.
    """

    def __init__(self) -> None:
        super().__init__()
        self.subscription = None
        self.subscribed_variables = {}
        self.event_node = None
        self.events: pd.DataFrame = None
        self.last_event_update = dt.datetime.now()

    async def subscribe_data_change(self, server: Server, nodes: list[BaseVariable], period: float = 500):
        """
        Subscribe to data change.

        Args:
            server: Server - the server object.
            nodes: list[BaseVariable] - the variables to subscribe to.
            period: float - the subscription period.
        """
        # make sure nodes are not none
        if len(nodes) == 0:
            return
        nodes = remove_none(nodes)
        if self.subscription is None:
            self.subscription = await server.client.create_subscription(period, self)
        self.subscribed_variables = dict((node.nodeid, node) for node in nodes)
        result = await self.subscription.subscribe_data_change(nodes)
        return result

    async def subscribe_events(self, server: Server, node: Node, period: float = 500):
        """
        Subscribe to events.

        Args:
            server: Server - the server object.
            node: Node - the event node.
            period: float - the subscription period.
        """

        if self.subscription is None:
            self.subscription = await server.client.create_subscription(period, self)
        self.event_node: LADSNode = node
        # asyncua subscribes events with queuesize=0 ("server default"); node-opcua revises that to 1, so of several events
        # raised within one publishing interval only one is delivered. A queue keeps all of them (asyncua 1.1.8 … 2.1.0).
        return await self.subscription.subscribe_events(node, queuesize=EVENT_QUEUE_SIZE)

    async def datachange_notification(self, node: Node, val: Any, data: DataChangeNotif):
        """
        Notification of a data change.

        Args:
            node: Node - the node.
            val: Any - the value.
            data: DataChangeNotif - the data change notification.
        """

        try:
            variable: Node = self.subscribed_variables[node.nodeid]
            variable.data_change_notification(data)
        except Exception as error:
            _logger.error(f"data_change_notification error {error}")

    async def event_notification(self, event: Event):
        """
        Notification of an event.

        Args:
            event: Event - the event.
        """

        # several events within one publishing interval arrive here one by one, as long as the event subscription has a
        # queue (subscribe_events: queuesize=EVENT_QUEUE_SIZE); with the asyncua default the server keeps only one of them
        fields_dict = event.get_event_props_as_fields_dict()
        event_fields = {}
        try:
            event_fields = {k: variant_value_to_str(v) for k, v in fields_dict.items()}
        except Exception as error:
            _logger.error(error)
        print(event_fields["Time"], event_fields["SourceName"], event_fields["Message"])
        key = pd.to_datetime(dt.datetime.now())
        if self.events is None:
            self.events = pd.DataFrame(event_fields, index = [key])
        else:
            self.events.loc[key] = event_fields
            if len(self.events.index) > 1000:
                self.events = self.events.tail(-10)
        self.last_event_update = key

    async def status_change_notification(self, status: Any):
        """
        Notification of a status change.

        Args:
            status: Any - the status.
        """

        print(status)
