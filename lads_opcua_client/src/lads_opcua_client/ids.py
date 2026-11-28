"""Node ids of the LADS, Machinery and DI namespaces."""

from __future__ import annotations

from enum import IntEnum


# MARK: Node IDs
# pylint: disable=C0103
class LADSObjectIds(IntEnum):
    """LADS specific numerical node-ids"""
    DeviceType = 1002
    ComponentSetType = 1025
    ComponentType = 1024
    SetType = 61
    FunctionalUnitSetType = 1023
    FunctionalUnitType= 1003
    FunctionSetType = 1026
    FunctionType= 1004
    AnalogScalarSensorFunctionType = 1016
    AnalogScalarSensorFunctionWithCompensationType = 1000
    AnalogArraySensorFunctionType = 1015
    TwoStateDiscreteSensorFunctionType = 1031
    MultiStateDiscreteSensorFunctionType = 1037
    AnalogControlFunctionType = 1009
    AnalogControlFunctionWithTotalizerType = 1014
    AnalogControlFunctionWithComposedTargetValueType = 1052
    AnalogControlFunctionWithRelativeTargetValueType = 1029
    TimerControlFunctionType = 1013
    TwoStateDiscreteControlFunctionType = 1042
    MultiStateDiscreteControlFunctionType = 1045
    MulitModeControlFunctionType = 1047
    ControllerParameterType = 1048
    ControllerParameterSetType = 1049
    StartStopControlFunctionType = 1032
    CoverFunctionType = 1011
    ProgramManagerType = 1006
    ProgramTemplateSetType = 1019
    ProgramTemplateType = 1018
    ActiveProgramType = 1040
    ResultSetType = 1020
    ResultType = 1021
    ResultFileSetType = 1022
    ResultFileType = 1001
    VariableSetType = 1041


class MachineryObjectIds(IntEnum):
    """Machinery specific numerical node-ids"""
    MachineryItemIdentificationType = 1004
    MachineryOperationCounterType = 1009
    MachineryLifeTimeCounterType = 1015


class DIObjectIds(IntEnum):
    """DI specific numerical node-ids"""
    LockingServicesType = 6388
    MaxInactiveLockTime = 6387
    DeviceHealthEnumeration = 6244
    LifetimeVariableType = 468


# MARK: SubscriptionLevel
class SubscriptionLevel(IntEnum):
    """Subscription levels for variables"""
    Never = 0
    Temporary = 1
    Permanent = 2
