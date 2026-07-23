# HELM api. Build from the REPO ROOT, not from backend/:
#   docker build -f backend/Dockerfile -t brain40/helm-api:0.1.0 .
# (House convention. Every Dockerfile here COPYs with a leading dir so one
# context serves them all — building from the wrong dir fails on the COPY.)
FROM python:3.12-slim

WORKDIR /app

# Dependencies first so a source-only change doesn't reinstall the world.
COPY backend/pyproject.toml ./pyproject.toml
COPY backend/src ./src

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# Where SQLite + jwt_secret + auth.json live. Bind-mounted in compose from a
# host directory you choose — nothing here is disposable.
ENV HELM_DATA_DIR=/data
VOLUME ["/data"]

EXPOSE 8000

# Process liveness only. Deliberately does NOT check Lemonade: HELM restarting
# cannot fix a dead lemond, and a healthcheck that conflates the two turns an
# upstream outage into a restart loop.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status==200 else 1)"

# --proxy-headers / --forwarded-allow-ips: HELM sits behind nginx behind NPM.
# Without these, uvicorn builds redirect URLs from the internal http hop and
# can bounce an https client down to http; the redirect cascade then drops the
# POST body and surfaces as a baffling 405. URSA 1.1.6 lost a session to
# exactly that. Related: the session cookie's Secure flag keys off the
# X-Forwarded-Proto these make visible — see auth/routes.py::_detect_scheme.
#
# (Comments must sit ABOVE the exec-form CMD, never inside the JSON array.)
CMD ["uvicorn", "kblocallm.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
