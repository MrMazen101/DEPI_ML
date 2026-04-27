import streamlit as st
from config import Config
from main import TrainingPipeline

# 1. Dashboard Title
st.title("🧠 Neural Network Training Dashboard")
st.write("This interface controls the model's settings and starts the training with the click of a button.")

# 2. Sidebar for Config Management
st.sidebar.header("⚙️ Hyperparameters")

# Linking the UI inputs directly to your Config file
Config.EPOCHS = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=Config.EPOCHS)
Config.BATCH_SIZE = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
Config.LEARNING_RATE = st.sidebar.number_input("Learning Rate", value=Config.LEARNING_RATE, format="%.4f")

# 3. Display Current Settings
st.subheader("Current Experiment Settings:")
st.info(f"The model will train for **{Config.EPOCHS}** epochs, processing **{Config.BATCH_SIZE}** images at a time.")

# 4. Execution Button
if st.button("🚀 Start Training Now"):
    with st.spinner("The model is learning and training... Check the VS Code Terminal below for live details."):
        # Running your pipeline maestro from here
        pipeline = TrainingPipeline()
        pipeline.run()
        
    st.success("Training finished successfully! 🎉 The model is ready.")