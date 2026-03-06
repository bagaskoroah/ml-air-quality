# docker container will run in python version: 3.12-slim
FROM python:3.12-slim

# set working directory to execute the program
WORKDIR /home

# copy anything needed to docker container
COPY ./requirements.txt ./
COPY ./src ./src/
COPY ./config ./config/

# execute when building docker images
RUN apt-get update && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# expose port the app runs on
EXPOSE 8501

# execute when running docker container
CMD ["streamlit", "run", "src/ui.py"]