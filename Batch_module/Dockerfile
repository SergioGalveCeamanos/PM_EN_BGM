# Use the Python3.7.2 image
FROM python:3.7

# Set the working directory to /app
WORKDIR /batch_module

# Copy the current directory contents into the container at /app 
ADD . /batch_module

# Install the dependencies
RUN pip install -r requirements.txt

#RUN pip3 install uwsgi

# run the command to start uWSGI
#CMD ["uwsgi", "app.ini"]

# run the command to start the loop
CMD [ "python", "./app.py" ]