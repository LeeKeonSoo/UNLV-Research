FROM python:3.12-slim

RUN pip install --no-cache-dir datasets==3.6.0 numpy==2.2.6 pandas==2.2.3 tqdm==4.67.1

COPY lcb_runner /opt/livecodebench/lcb_runner
COPY .git/HEAD /opt/livecodebench/.git/HEAD

ENV PYTHONPATH=/opt/livecodebench
WORKDIR /work

ENTRYPOINT ["python", "/work/livecodebench_evaluate.py"]
