# Phase 1: Define the Base Image
# We start with an official Python image. 'slim' is a good choice as it's smaller.
FROM python:3.9-slim

# Phase 2: Set up the Environment
# This is the directory inside the container where our app will live.
WORKDIR /app

# Phase 3: Install Dependencies
# Copy the requirements file first to leverage Docker's layer caching.
# If the requirements don't change, Docker won't reinstall them on every build.
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Phase 4: Copy the Application Code
# Copy all the files from your project folder into the container's /app directory.
COPY . .

# Phase 5: Expose the Port
# Tell Docker that the container listens on port 5000 (the default for Flask).
EXPOSE 5000

# Phase 6: Define the Run Command
# This is the command that starts the Flask application when the container runs.
# It's equivalent to running 'python app.py' in your terminal.
CMD ["python", "app.py"]
