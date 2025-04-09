""" Command Line Interface for Starting the LADS OPC UA Viewer"""

#import subprocess
import os

def main():
    try:
        app_path = os.path.join(os.path.dirname(__file__), "main.py")
        os.system(f"streamlit run {app_path}")
    except KeyboardInterrupt:
        print("Streamlit app interrupted!")
    except Exception as e:
        print(f"Error: {e}")
    #subprocess.run(["streamlit", "run", app_path]) # Alternative to os.system
