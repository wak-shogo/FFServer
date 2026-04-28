import streamlit as st
import json
import os
import pandas as pd

# Path to the queue file
PROJECTS_DIR = "simulation_projects"
QUEUE_FILE = os.path.join(PROJECTS_DIR, "queue.json")

st.set_page_config(page_title="Queue Manager", page_icon="📋", layout="centered")

st.title("📋 Job Queue Manager")
st.markdown("Use this tool to remove jobs from the waiting list without stopping the server.")

def load_queue():
    if not os.path.exists(QUEUE_FILE):
        return []
    try:
        with open(QUEUE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading queue: {e}")
        return []

def save_queue(queue_data):
    try:
        with open(QUEUE_FILE, 'w') as f:
            json.dump(queue_data, f, indent=4)
        st.success("Queue updated successfully!")
    except Exception as e:
        st.error(f"Error saving queue: {e}")

# Load current queue
queue = load_queue()

if not queue:
    st.info("The queue is currently empty.")
else:
    st.write(f"**Current Queue Length:** {len(queue)}")
    
    # Create a simplified display list for the dataframe
    display_data = []
    for i, job in enumerate(queue):
        display_data.append({
            "Index": i,
            "Project Name": job.get("project_name", "N/A"),
            "Model": job.get("model", "N/A"),
            "Type": job.get("job_type", "N/A"),
            "Original File": job.get("original_filename", "N/A")
        })
    
    df = pd.DataFrame(display_data)
    
    # Display as a table
    st.dataframe(df.set_index("Index"), use_container_width=True)

    st.subheader("🗑️ Delete Jobs")
    
    # Using a form to group the selection and button
    with st.form("delete_form"):
        # Multi-select for deletion using the index
        selected_indices = st.multiselect(
            "Select jobs to REMOVE:",
            options=df["Index"].tolist(),
            format_func=lambda x: f"#{x}: {df.iloc[x]['Project Name']} ({df.iloc[x]['Model']})"
        )
        
        submitted = st.form_submit_button("Remove Selected Jobs", type="primary")
        
        if submitted:
            if not selected_indices:
                st.warning("Please select at least one job to remove.")
            else:
                # Filter out the selected indices
                # We check if 'i' is NOT in 'selected_indices' to keep it
                new_queue = [job for i, job in enumerate(queue) if i not in selected_indices]
                
                save_queue(new_queue)
                st.rerun()

    st.divider()
    with st.expander("Advanced: Raw JSON View"):
        st.json(queue)
