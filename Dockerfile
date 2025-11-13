# Source: https://gemini.google.com/share/7e8c257f10f3

# Use a base image with Python
FROM python:3.12.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for the python packages (like sentence-transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies first
# This allows Docker to cache the layer if requirements.txt doesn't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code
COPY codes codes/
COPY data data/

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# The application now relies on a separate Ollama service running on the network.
# Modify the entrypoint to run the Streamlit app.
# The 'streamlit_board_game_qa.py' code will need to be modified 
# to remove the start_server/stop_server calls, as Ollama will be running
# in its own service.
CMD ["streamlit", "run", "codes/streamlit_board_game_qa.py", "--server.port=8501", "--server.address=0.0.0.0"]