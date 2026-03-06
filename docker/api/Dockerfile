# docker container will run in python version: 3.12-slim
FROM python:3.12-slim

# set working directory to execute the program
WORKDIR /home

# copy anything needed to docker container
COPY ./requirements.txt ./
COPY ./src ./src/
COPY ./models ./models/
COPY ./config ./config/

# execute when building docker images
RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# expose port the app runs on
EXPOSE 8080

# execute when running docker container
CMD ["python", "src/api.py"]