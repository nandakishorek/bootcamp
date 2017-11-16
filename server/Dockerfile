FROM ubuntu:16.04
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python-pip && \
    pip3 install Flask && \
    pip3 install pandas && \
    pip3 install scipy
ADD src /opt/src
ADD data /opt/data
WORKDIR /opt/src
ENV FLASK_APP=main.py
ENV FLASK_DEBUG=1
EXPOSE 5000
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]
