# Use the Python3.7.2 image
FROM python:3.7

# Set the working directory to /app
WORKDIR /pm_manager

# Copy the current directory contents into the container at /app 
ADD . /pm_manager

# Install the dependencies
RUN pip install -r requirements.txt

#RUN pip3 install uwsgi

RUN pip3 install schedule

# run the command to start uWSGI
#CMD ["uwsgi", "app.ini"]

# run the command to start the loop
CMD [ "python", "./app.py" ]