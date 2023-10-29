FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade -r requirements.txt

EXPOSE 8503

# HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health

ENTRYPOINT ["python", "app.py" ]