name: Training Pipeline CI/CD

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/
      - name: Build and push Docker image
        run: |
          docker build -t room_occupancy_model .
          docker tag room_occupancy_model:latest <dockerhub-username>/room_occupancy_model:latest
          docker push <dockerhub-username>/room_occupancy_model:latest
