import streamlit as st

def getOutput_drafts(planner_response, writer_response, editor_response):
    st.header("Agent Draft Responses")
    st.write(":gray[The draft of response by each agent]")
    
    with st.expander("Planner Draft", expanded=False):
        st.subheader("Planner Draft")
        if planner_response == "":
            st.write(":gray[Generate an article to get planner response.]")
        else:
            st.markdown(planner_response)
            
    with st.expander("Writer Draft", expanded=False):
        st.subheader("Writer Draft") 
        if writer_response == "":
            st.write(":gray[Generate an article to get writer response.]")
        else:
            st.markdown(writer_response)
        
    with st.expander("Editor Draft", expanded=False):
        st.subheader("Editor Draft")
        if editor_response == "":
            st.write(":gray[Generate an article to get editor response.]")
        else:
            st.markdown(editor_response)
    