FROM python:3.9
RUN pip install requests datetime pandas matplotlib scikit-learn
COPY . /app
WORKDIR /app
CMD ["python", "Music Recommender System Final Project.py"]
