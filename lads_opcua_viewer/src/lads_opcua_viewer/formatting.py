"""Formatting of values and descriptions for display."""

import lads_opcua_client as lads


# MARK: format_value
def format_value(x: float | list[float], decis = 1) -> str:
    result = "NaN"
    try:
        if isinstance(x, list):
            result = f"[{format_number(x[0], decis)} .. {format_number(x[len(x) - 1], decis)}]"
        else:
            result = format_number(x, decis)
    finally:
        return result


# MARK: format_analog_item
def format_analog_item(item: lads.AnalogItem, include_eu = False) -> str:
    result = "NaN"
    try:
        decis = item.decimals()
        eu = item.eu
        value = item.value

        def scale(v: float) -> float:
            return 0.001 * v if eu == "s" else v

        if isinstance(value, list):
            scaled = [scale(v) for v in value]
            number = format_value(scaled, decis)
        else:
            number = format_number(scale(value), decis)

        result = f"{number} {eu}" if include_eu else number
    except Exception as e:
        print(e)
    finally:
        return result


# MARK: format_number
def format_number(x: float, decis = 1) -> str:
    result = "NaN"
    try:
        result = f"{{0:.{decis}f}}".format(x)
    finally:
        return result


def description_text(node) -> str:
    """Description (LocalizedText) of a node as plain text, empty if missing."""
    description = getattr(node, "description", None)
    text = getattr(description, "Text", None) if description is not None else None
    return text.strip() if text else ""


def help_markdown(variable: lads.BaseVariable, function: lads.LADSNode = None, alarm_limits: str = "") -> str | None:
    """Help text of a value widget: description of the function (or the variable if the function has none),
    the ontology definitions of the variable and, if present, the alarm limits."""
    parts: list[str] = []
    description = description_text(function) or description_text(variable)
    if description:
        parts.append(description)
    if variable is not None and variable.dictionary_entries_as_markdown:
        parts.append(variable.dictionary_entries_as_markdown)
    if alarm_limits:
        parts.append(alarm_limits)
    return "  \r\n".join(parts) if parts else None


# MARK: to_float
def to_float(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except:
        result = default
    return result
