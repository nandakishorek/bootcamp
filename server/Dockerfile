FROM ubuntu:16.04
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    pip3 install Flask && \
    pip3 install -U flask-cors && \
    pip3 install pandas && \
    pip3 install scipy
ADD src /opt/src
ADD data /opt/data
WORKDIR /opt/src
ENV FLASK_APP=main.py
ENV FLASK_DEBUG=1
EXPOSE 5000
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
