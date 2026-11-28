"""View of the charts of a functional unit."""

import lads_opcua_client as lads
import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import time
from asyncua import ua
from typing import Tuple
from lads_opcua_viewer.formatting import format_number


# MARK: add_chart_items
def add_chart_items(functions: list[lads.Function], traces: list, arrays: list):
    for function in functions:
        analog_item: lads.AnalogItem = None
        if isinstance(function, lads.AnalogControlFunction):
            analog_item = function.current_value
        elif isinstance(function, lads.AnalogScalarSensorFunction):
            analog_item = function.sensor_value
        if analog_item is not None:
            if isinstance(analog_item.value, list):
                arrays.append((function, analog_item))
            elif analog_item.history is not None:
                traces.append((function, analog_item))
        # recurse sub-functions
        if function.function_set is not None:
            add_chart_items(function.function_set.functions, traces=traces, arrays=arrays)


# MARK: update_charts
def update_charts(container, functional_unit: lads.FunctionalUnit, use_plotly=True):
    with container.container():
        # collect analog items with history and arrays
        traces: list[Tuple[lads.Function, lads.AnalogItem]] = []
        arrays: list[Tuple[lads.Function, lads.AnalogItem]] = []
        idx = 0

        add_chart_items(functional_unit.functions, traces=traces, arrays=arrays)

        if use_plotly:
            # add traces
            fig = go.Figure()
            layout_dict = {}
            pos_left = 0.1
            pos_right = 0.9
            layout_dict["xaxis"] = dict(domain=[pos_left, pos_right])
            index = 0
            count = len(traces)
            for trace in traces:
                index += 1
                function, analog_item = trace
                df: pd.DataFrame = analog_item.history
                fig.add_trace(go.Scatter(
                    x = df.index.array,
                    y = df.iloc[:, 0].array,
                    name = function.display_name,
                    yaxis = f"y{index}"
                ))
                color="#404040"
                yaxis_key = "yaxis" if index <= 1 else f"yaxis{index}"
                left  = index <= count / 2
                side = "left" if left else "right"
                position = pos_left - 0.2 * (index - 1) if left else pos_right + 0.2 * ((index - 1) - count / 2)
                position = 0 if position < 0 else 1 if position > 1 else position
                yaxis_dict = dict(
                    title = dict(
                        text = f"{function.display_name} [{analog_item.eu}]",
                        font = dict(
                            color = color
                        )
                    )
                )
                if index > 1:
                    yaxis_dict["anchor"] = "x"
                    yaxis_dict["overlaying"] = "y"
                    yaxis_dict["side"] = side
                    yaxis_dict["position"] = position

                layout_dict[yaxis_key] = yaxis_dict


            fig.update_layout(layout_dict)

            fig.update_layout(
                # title_text = functional_unit.display_name,
                width = 1000,
                # title_x = 0.1,
                legend = dict(yanchor = "top", xanchor = "left", x = 0.7, y = 1.35),
            )
            with st.expander("**Chart**", expanded=(idx==0)):
                unique_key = f"plotly_chart_{time.time_ns()}"
                #st.plotly_chart(fig, use_container_width=True, key=unique_key)
                st.plotly_chart(fig, width="stretch", key=unique_key)
                idx += 1
        else:
            for trace in traces:
                function, analog_item = trace
                with st.expander(f"**Chart {function.display_name}**", expanded=(idx==0)):
                    unique_key = f"line_chart_{time.time_ns()}"
                    st.line_chart(data = analog_item.history, height = 100)
                    idx += 1

        # tables
        for array in arrays:
            function, analog_item = array
            value = analog_item.value
            i = len(value)
            col_count = int(math.sqrt(3 / 2 * i))
            row_count = int(2 / 3 * col_count)
            isPlate = col_count * row_count == i
            with st.expander(f"**{function.display_name}**", expanded=(idx<=2)):
                if isPlate:
                    cols: dict = {}
                    cols["Plate"] = list(range(1, row_count + 1))
                    # cols[f"**{function.display_name}**"] = list(range(1, row_count + 1))
                    for col_idx in range(col_count):
                        values = []
                        for row_idx in range(row_count):
                            values.append(format_number(value[col_idx * row_count + row_idx]))
                        cols[chr(ord("A") + col_idx)] = values

                    df = pd.DataFrame(cols)
                    if analog_item.eu_range is not None:
                        eu_range: ua.Range = analog_item.eu_range
                        try:
                            df.style.background_gradient(
                                axis=None,
                                vmin=eu_range.Low,
                                vmax=eu_range.High,
                                cmap="jet"
                            )
                        except AttributeError:
                            ' do nothing'
                    # st.dataframe(df, use_container_width=True, hide_index=True)
                    st.dataframe(df, width="stretch", hide_index=True)
                else:
                    eu = analog_item.engineering_units
                    col = eu.DisplayName.Text if (eu is not None and eu.DisplayName is not None and eu.DisplayName.Text) else "y"
                    df = pd.DataFrame({col: value})
                    unique_key = f"area_chart_{time.time_ns()}"
                    st.area_chart(df)
            idx += 1
