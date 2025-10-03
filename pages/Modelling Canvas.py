import streamlit as st

try:
    # Import the canvas app and run its main()
    import src.plexos.modelling_canvas.canvas_app as canvas_app
except Exception as e:
    st.error(f"Failed to load modelling canvas: {e}")
else:
    st.title("Modelling Canvas")
    # Delegate to the canvas app
    canvas_app.main()
