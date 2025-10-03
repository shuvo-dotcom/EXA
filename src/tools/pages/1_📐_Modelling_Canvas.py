import streamlit as st

try:
    # Import the canvas app and run its main()
    from src.plexos.modelling_canvas import canvas_app
except Exception as e:
    st.error(f"Failed to load modelling canvas: {e}")
else:
    st.title("Modelling Canvas")
    canvas_app.main()
