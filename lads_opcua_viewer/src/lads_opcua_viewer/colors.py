"""Colors of states and variable status in the viewer."""

import lads_opcua_client as lads


# MARK: function_state_color
def function_state_color(function: lads.BaseControlFunction) -> str:
    return state_color(function.current_state)


# MARK: variable_status_color
def variable_status_color(variable: lads.BaseVariable, color_good = 'blue') -> str:
    status_code = variable.data_value.StatusCode
    if status_code.is_bad():
        return 'red'
    elif status_code.is_uncertain():
        return 'orange'
    else:
        return color_good


# MARK: state_color
def state_color(current_state: lads.BaseVariable) -> str:
    s = str(current_state.value_str)
    return "green" if "Running" in s else "red" if "Abort" in s else "gray"
