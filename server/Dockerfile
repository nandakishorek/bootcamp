FROM ubuntu:16.04
RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python-pip && \
    pip install Flask
ADD src /opt/src
WORKDIR /opt/src
ENV FLASK_APP=main.py
ENV FLASK_DEBUG=1
EXPOSE 5000
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
