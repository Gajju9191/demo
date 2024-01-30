FROM ubuntu:latest
FROM nginx:1.19.10
FROM python:3.10.6

# set a directory for the app
WORKDIR /usr/src/bootapp
COPY . /usr/src/bootapp/

# using the default streamlit port number for container to expose
EXPOSE 8501

# install dependencies
RUN apt-get update
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 app.py
# running the streamlit app
ENTRYPOINT ["streamlit", "run", "app.py"]
