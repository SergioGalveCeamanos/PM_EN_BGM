# Use the Python3.7.2 image
FROM python:3.7

# Set the working directory to /app
WORKDIR /db_manager

# Copy the current directory contents into the container at /app 
ADD . /db_manager

# Install the dependencies
RUN pip install -r requirements.txt

RUN pip3 install uwsgi

# run the command to start uWSGI
CMD ["uwsgi", "app.ini"]