"""View of compliance documents (including DCC documents)."""

import lads_opcua_client as lads
import pandas as pd
import streamlit as st
import time
from lads_opcua_viewer.dcc.dcc_md import generate_markdown_from_dcc_xml


# MARK: show_documents
def show_documents(container, document_set: lads.LADSSet):
    with container.container():
        container_documents, container_document_detail = st.columns([0.25, 0.75])
        with container_documents:
            if st.button("Refresh"):
                pass
    render_documents(container_documents, container_document_detail, document_set)
    return container_documents, container_document_detail


def update_documents(container_documents, container_document_detail, document_set: lads.LADSSet):
    pass


def render_documents(container_documents, container_document_detail, document_set: lads.LADSSet):

    def append(list: list[str], item: lads.BaseVariable, default = ""):
        list.append(item.value_str if item is not None else default)

    # prepare data
    documents: list[lads.ComplianceDocument] = document_set.children
    name = []
    document_name = []
    issued_at = []
    valid_from = []
    valid_until = []
    mime_type = []
    content = []
    schema_uri = []
    for document in documents:
        name.append(document.display_name)
        append(document_name, document.document_name)
        append(issued_at, document.issued_at)
        append(valid_from, document.valid_from)
        append(valid_until, document.valid_until)
        append(mime_type, document.mime_type)
        append(content, document.content)
        append(schema_uri, document.schema_uri)
    data = pd.DataFrame({"Name": document_name, "Issued At": issued_at,
                         "Valid From": valid_from, "Valid Until": valid_until,
                         "Content": content, "Mime Type": mime_type, "Schema URI": schema_uri })
    # build view
    with container_documents:
        unique_key = f"select_document"
        selected_name = st.radio("Select a document", options=data["Name"], key=unique_key, index=0)
    with container_document_detail:
        if selected_name is None:
            st.write("Select a document")
        else:
            selected = data[data["Name"] == selected_name].iloc[0]
            d = data[data["Name"] == selected_name]
            st.dataframe(d, column_order=["Name", "Issued At", "Valid From", "Valid Until", "Mime Type", "Schema URI"], hide_index=True, key=unique_key)
            index = d.index.to_list()[0]
            document: lads.LADSNode = documents[index]
            st.markdown(f"***Ontology Definitions***  \r\n{document.dictionary_entries_as_markdown}")
            references: list[str] = document.references_markdown
            st.markdown(f"***References***  \r\n{"  \r\n".join(references)}")
            unique_key = f"document_table_{time.time_ns()}"
            mime_type: str = selected["Mime Type"]
            content: str = selected["Content"]
            if content is not None:
                if mime_type.find("text") >= 0:
                    st.text(content)
                elif mime_type.find("json") >= 0:
                    st.json(content)
                elif mime_type.find("xml") >= 0:
                    if mime_type.find("dcc") >= 0:
                        tab_md, tab_xml = st.tabs(["Human Readable", "Raw XML"])
                        with tab_md:
                            md = generate_markdown_from_dcc_xml(content)
                            st.markdown(md)
                        with tab_xml:
                            st.code(content, language="xml", height=500)
                    else:
                        st.code(content, language="xml")
            if mime_type.find("pdf") >= 0:
                try:
                    if not document.has_data():
                        document.fetch_data()
                        st.text("Loading PDF ..")
                    pdf = document.data
                    if pdf is not None:
                        unique_key = f"document_pdf_{time.time_ns()}"
                        st.pdf(pdf, key=unique_key)
                except Exception as err:
                    st.error(err)
