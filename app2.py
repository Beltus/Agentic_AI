import streamlit as st
import hashlib
from typing import List, Dict
import os

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants, settings
from utils.logging import logger



def main():

    processor = DocumentProcessor() # initialize document processor to extract structured content from uploaded files
    retriever_builder = RetrieverBuilder() #initialize hybrid retriever (BM25 + VectorSearch)
    workflow = AgentWorkflow() # Initialize workflow to orchestrate the multi-agent processing pipeline using LangGraph.

    # Page config
    st.set_page_config(page_title="ChattyDoc", layout="wide")

    # ---- TOP NAVBAR ----
    st.markdown(
        """
        <style>
        .top-bar-container {
            background-color: #000;
            padding: 10px;
            color: white;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
        .main > div {
            margin-top: 10px;
        }
        </style>
        <div class="top-bar-container">ChattyDoc</div>
        """,
        unsafe_allow_html=True,
    )

    # ---- CUSTOM CSS FOR BORDERED COLUMNS ----
    # ---- SPECIFIC CSS FOR 3 MAIN COLUMNS ONLY ----
    st.markdown(
        """
        <style>
        /* 1. Target only the columns within the main row */
        /* This prevents individual widgets from getting borders */
        [data-testid="column"] {
            border: 1px solid #31333F; 
            padding: 25px;
            border-radius: 15px;
            background-color: #0E1117;
            min-height: 80vh; /* Optional: Keeps columns same height */
        }

        /* 2. Remove any default borders Streamlit might add to internal containers */
        [data-testid="stVerticalBlock"] > div:has(div[data-testid="stVerticalBlock"]) {
            border: none !important;
            box-shadow: none !important;
        }

        /* 3. Adjust the header inside these specific boxes */
        [data-testid="column"] h3 {
            font-size: 1.3rem;
            margin-top: 0px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state

    if "retriever" not in st.session_state:
        st.session_state.update({
            "file_hashes": frozenset(),
            "retriever": None
        })

    if "verification_text" not in st.session_state:
        st.session_state.update({
                "final_answer": " ",
                "verification_text": ""
            }
        )

    # 5) Standard flow for question submission
    def process_question(question_text: str, uploaded_files: List, state: Dict):
        """Handle questions with document caching."""
        # 1) Validate inputs
        if not question_text.strip():
            st.error("❌ Please enter a question")
            st.stop()
        if not uploaded_files:
            st.error("❌ Please upload atleast one document")
            st.stop()
        # if not selected:
        #     st.error("❌ Please select at least one source")
        #     st.stop()
    
        #compute unique hashes for all uploaded files.
        current_hashes = _get_file_hashes(uploaded_files) 

        # 4) Rebuild retriever if needed
        if (st.session_state.retriever is None or current_hashes != st.session_state.file_hashes):
            logger.info("Building new retriever...")
            chunks = processor.process(uploaded_files)

            #Build retriever object
            new_retriever = retriever_builder.build_hybrid_retriever(chunks)

            retriever = new_retriever

            st.session_state["retriever"] = new_retriever
            st.session_state["file_hashes"] = current_hashes

        else:
            retriever = st.session_state.retriever

        # 4. FINAL SAFETY GUARD
        if retriever is None:
            st.error("Retriever could not be initialized. Please try refreshing.")
            st.stop()

        # BACKEND PIPELINE - Execute pipeline
        with st.spinner("Agents are thinking..."):
            result = workflow.full_pipeline(
                question=question_text,
                retriever=retriever
            )
        return result["draft_answer"], result["verification_report"], state
    
    # ---- MAIN LAYOUT ----
    col1, col2, col3 = st.columns((2, 3, 2), gap = "medium")

    # =========================
    # ---- SOURCES COLUMN ----
    # =========================

    with col1:
        with st.container(border=True):
            st.subheader("Sources")

            # Upload button
            uploaded_files = st.file_uploader(
                "+ Add sources", accept_multiple_files=True, type=["pdf", "docx", "txt"]
            )

            # List of uploaded sources
            st.markdown("### Uploaded sources")

            # Track selected sources
            #selected = []

            # if uploaded_files:
            #     for file in uploaded_files:
            #         checkbox = st.checkbox(file.name)
            #         if checkbox:
            #             selected.append(file.name)

    # =========================
    # ---- CHAT COLUMN ----
    # =========================

    # ---- Chat Column ----

    with col2:
        with st.container(border=True):

            st.subheader("Chat")

            # Create two columns: 80% for input, 20% for button
            input_col, btn_col = st.columns([4, 1])

            with input_col:
                question_text = st.text_input(
                    "What is your question?", 
                    key="question_input",
                    placeholder="Type your question here...",
                    label_visibility = "hidden"
                )

            with btn_col:
                # This adds the exact vertical space needed to align with a labeled text_input
                st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
                submit_btn = st.button("Enter", use_container_width=True)

            #when user clicks Enter button
            if submit_btn:
                
                #Run the workflow to process document and return answer and verification, state
                ai_response, verif, new_state = process_question(question_text, uploaded_files, st.session_state)

                #save results to session_state
                st.session_state.final_response = ai_response
                st.session_state.verification_text = verif

                #update the verification textbox and research agent response widget content to be displayed inside the verification box 
                # (Note: When you create a widget with key="verification_display", Streamlit automatically creates a corresponding entry in st.session_statewith that exact name. )
                st.session_state["verification_display"] = verif 
                st.session_state["research_agent_display"] = ai_response

                #Force rerun to make sure UI reflects changes immediately.
                st.rerun()

            if st.session_state.get("research_agent_display"):
                # Display Research agent response
                st.text_area(
                    label = "research_agent_message",
                    height = 200,
                    key = "research_agent_display",
                    label_visibility = "hidden"
                )

    # =========================
    # ---- SUMMARY COLUMN ----
    # =========================

    with col3:
        with st.container(border=True):

            st.subheader("Sources Summary")

            # Placeholder for summary
            summary_text = st.text_area(
                "", "summary of selected sources…", height=150
            )

            st.markdown("### Verification Report")
        
            # Display Verification agent output 
            st.text_area(
                label = "Verification",
                height = 200,
                key = "verification_display" 
            )
        
#Purpose: Ensures same file not loaded more than once
def _get_file_hashes(uploaded_files: List) -> frozenset:
    """Generate SHA-256 hashes for uploaded files."""
    hashes = set()
    for file in uploaded_files:
        # 1. Get the bytes directly from the Streamlit UploadedFile object
        file_bytes = file.getvalue()
        
        # 2. Hash the bytes
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        hashes.add(file_hash)
        
    return frozenset(hashes)

if __name__ == "__main__":
    main()
