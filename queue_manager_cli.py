import json
import os
import pandas as pd
import sys

# Constants
PROJECTS_DIR = "simulation_projects"
QUEUE_FILE = os.path.join(PROJECTS_DIR, "queue.json")

def load_queue():
    if not os.path.exists(QUEUE_FILE):
        return []
    try:
        with open(QUEUE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading queue: {e}")
        return []

def save_queue(queue_data):
    try:
        with open(QUEUE_FILE, 'w') as f:
            json.dump(queue_data, f, indent=4)
        print("✅ Queue updated successfully!")
    except Exception as e:
        print(f"❌ Error saving queue: {e}")

def main():
    print("\n========================================")
    print("      🧪 Queue Manager (CLI)      ")
    print("========================================\n")

    while True:
        queue = load_queue()

        if not queue:
            print("The queue is currently empty.")
            print("\n----------------------------------------")
            choice = input("Press Enter to refresh or 'q' to quit: ")
            if choice.lower() == 'q':
                break
            continue

        # Prepare data for display
        display_data = []
        for i, job in enumerate(queue):
            display_data.append({
                "Index": i,
                "Project Name": job.get("project_name", "N/A"),
                "Model": job.get("model", "N/A"),
                "Type": job.get("job_type", "N/A")
            })
        
        df = pd.DataFrame(display_data)
        print(f"Current Job Queue ({len(queue)} jobs):")
        # Adjust pandas display to show full columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
        
        print("\n----------------------------------------")
        user_input = input("Enter [Index] to delete, 'r' to refresh, or 'q' to quit: ").strip()

        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 'r':
            continue
        
        try:
            idx_to_delete = int(user_input)
            if 0 <= idx_to_delete < len(queue):
                target_job = queue[idx_to_delete]
                print(f"\n⚠️  Are you sure you want to delete job #{idx_to_delete}?")
                print(f"   Project: {target_job.get('project_name')}")
                confirm = input("   Type 'yes' to confirm: ").strip().lower()
                
                if confirm == 'yes':
                    # Reload queue to be safe against concurrent writes
                    current_queue = load_queue()
                    if len(current_queue) != len(queue):
                        print("⚠️  Queue changed externally. Please retry.")
                        continue
                    
                    # Remove
                    del current_queue[idx_to_delete]
                    save_queue(current_queue)
                else:
                    print("Cancelled.")
            else:
                print("❌ Invalid index.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Ensure we are in the correct directory (FFServer) if running directly
    # But usually this script is run as `python FFServer/queue_manager_cli.py` from workspace
    # Fix paths if necessary
    if not os.path.exists(PROJECTS_DIR):
        # Try finding it relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        PROJECTS_DIR = os.path.join(script_dir, "simulation_projects")
        QUEUE_FILE = os.path.join(PROJECTS_DIR, "queue.json")
        
    main()
