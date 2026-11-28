"""Table of variables."""

import lads_opcua_client as lads
import pandas as pd
import streamlit as st


# MARK: show_variables_table
def show_variables_table(variables: list[lads.BaseVariable], has_description: bool = False):
    names = []
    values = []
    descriptions = []
    for variable in variables:
        names.append(variable.display_name)
        values.append(variable.value_str)
        if has_description:
            descriptions.append(variable.description.Text if variable.description.Text is not None else "")
    data: pd.DataFrame = {"Name": names, "Value": values, "Description": descriptions, } if has_description else {"Name": names, "Value": values}
    column_config={
        "Name": st.column_config.Column(
            None,
            help="Variable name",
            disabled=True,
        ),
        "Value": st.column_config.Column(
            None,
            help="Variable value",
            disabled=True,
        ),
        "Description": st.column_config.Column(
            None,
            help="Variable description",
            disabled=True,
        ),
    }
    # st.dataframe(data, use_container_width=True, hide_index=True, column_config=column_config)
    st.dataframe(data, width="stretch", hide_index=True, column_config=column_config)
