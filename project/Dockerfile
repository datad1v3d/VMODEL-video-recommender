FROM python

WORKDIR /app

COPY . /app

RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "mlfile.py"]