FROM python:3.9

WORKDIR /app

COPY . /app 

# RUN pip install fastapi uvicorn scikit-learn numpy python-multipart torch

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=./requirements.txt,target=./requirements.txt \
    pip install -r ./requirements.txt

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
