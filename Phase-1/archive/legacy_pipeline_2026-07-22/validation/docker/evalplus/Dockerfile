FROM python:3.12-slim

ARG EVALPLUS_VERSION=0.3.1

RUN pip install --no-cache-dir "evalplus==${EVALPLUS_VERSION}" \
    && python -c "from evalplus.data import get_human_eval_plus, get_mbpp_plus; get_human_eval_plus(); get_mbpp_plus()"

COPY prevalidate.py /opt/evalplus-prevalidation/prevalidate.py

ENV EVALPLUS_MAX_MEMORY_BYTES=1073741824

ENTRYPOINT ["python", "/opt/evalplus-prevalidation/prevalidate.py"]
