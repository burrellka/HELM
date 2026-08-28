# Ingress — Nginx Proxy Manager + Cloudflare mTLS

Spec §2: `helm.example.com`, NPM + Cloudflare, **mTLS required**.

This is the layer you cannot test from a laptop, and the layer where the log
stream will die in a way that looks like an application bug. Read §3 before
you build the log pane, not after.

---

## 1. The proxy host

NPM → **Hosts → Proxy Hosts → Add Proxy Host**

| Field | Value |
|---|---|
| Domain Names | `helm.example.com` |
| Scheme | `http` |
| Forward Hostname / IP | `helm-web` (or the host IP if NPM isn't on `helm-net`) |
| Forward Port | `80` |
| Cache Assets | **off** — the SPA's nginx already sets its own cache headers, and NPM caching on top is how you serve a stale `index.html` after a deploy |
| Block Common Exploits | on |
| Websockets Support | on (harmless; free if HELM ever surfaces Lemonade's :9000 realtime API) |

**SSL tab:** request a Let's Encrypt cert, **Force SSL** on, HTTP/2 on.

> If NPM is not on `helm-net`, add it — `docker network connect helm-net
> nginx-proxy-manager`. Forwarding to the host IP works too, but then HELM's
> port is reachable from the LAN, which is the exposure discussed in §5.

---

## 2. Cloudflare mTLS

Client-certificate auth happens at Cloudflare's edge, not in NPM.

1. **SSL/TLS → Client Certificates** → create one per device that should reach
   HELM. Install the `.pem` on those devices.
2. **SSL/TLS → Client Certificates → Hosts** → enable for
   `helm.example.com`.
3. **Security → WAF → Custom rules** → add:
   ```
   (http.host eq "helm.example.com" and not cf.tls_client_auth.cert_verified)
   → Block
   ```
   Without this rule Cloudflare *requests* a client cert but still serves
   people who don't present one. The rule is what makes mTLS actually
   mandatory. This is the step that gets skipped.
4. Verify: `curl https://helm.example.com/healthz` from a machine
   with no client cert must be **blocked**, not 200.

**Read this bit:** mTLS protects the *public* path. It is not a perimeter.
Anything on your LAN, any container on the same Docker network, and the Docker host itself
reach `helm-web:80` directly without touching Cloudflare. That is why HELM has
its own operator login.

---

## 3. ★ The streaming config. Do this before you build the log pane.

Spec §5.6 (live log stream) and §4.2 (`pull` with `stream: true`) both stream.
Behind this ingress, streams fail **twice** by default, and both failures look
like your code:

1. **NPM's nginx buffers the response.** It holds the whole thing and flushes
   at the end. The UI shows nothing. Your backend logs look perfect.
2. **Cloudflare 524s after ~100s of silence.** It is an *idle* timer, not a
   duration limit. A log tail at 3am is silent for far longer than 100s, so a
   working stream gets killed for being calm.

HELM's own `frontend/nginx.conf` already handles the container-side nginx.
**NPM has its own nginx in front of it and needs the same treatment**, and
there is no checkbox for it.

**Proxy Host → Advanced → Custom Nginx Configuration**, paste:

```nginx
# HELM: streaming endpoints (log tail, streaming pulls).
# Without proxy_buffering off, nginx holds the stream and the browser sees
# nothing until Cloudflare 524s it at ~100s of silence.
location /api/v1/logs/ {
    proxy_pass http://helm-web:80;
    proxy_http_version 1.1;              # chunked transfer requires 1.1
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;   # session cookie needs this
    proxy_set_header Connection "";

    proxy_buffering off;                 # THE line that matters
    proxy_cache off;
    proxy_read_timeout 3600s;            # default 60s would kill a long tail
    proxy_send_timeout 3600s;
}

# Model loads take 10-20s and pulls take minutes; the default 60s read timeout
# would fail a request that actually succeeded — the worst lie a control plane
# can tell, because the operator retries and the second load evicts something.
location /api/v1/ {
    proxy_pass http://helm-web:80;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Connection "";

    proxy_buffering off;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
}
```

The app side is already done for you:

- the backend sets `X-Accel-Buffering: no` + `Cache-Control: no-cache` on
  streaming responses (belt to nginx's suspenders);
- it emits a **keepalive line every ~15s** on a quiet stream — this is the one
  mechanism that actually defeats 524, because it guarantees a byte crosses the
  edge regardless of what the log is doing;
- the frontend reads with `fetch` + `ReadableStream` and drops `:`-prefixed
  keepalives (`api.streamLines` in `frontend/src/api/client.ts`).

**No Cloudflare "enable streaming" toggle exists.** Cloudflare passes a stream
through fine once the origin stops buffering. If it still buffers after the
above, check that no Cloudflare **Cache Rule** matches `/api/` — a cache rule
will absorb a stream.

Full background on the streaming/keepalive mechanics: `docs/ARCHITECTURE.md`
(the "Streaming" section) and HELM's own `frontend/nginx.conf`.

---

## 4. Verifying it end to end

```bash
# 1. Reachable, and mTLS is actually enforced.
curl -sS https://helm.example.com/healthz            # -> blocked (no client cert)
curl -sS --cert client.pem https://helm.example.com/healthz
# -> {"ok":true,"version":"0.9.12"}

# 2. Streaming is NOT buffered. Lines must appear as they happen. If they all
#    arrive at once at the end, something in the chain is still buffering.
curl -N --cert client.pem https://helm.example.com/api/v1/logs/stream

# 3. The 524 test — the one people skip. Open the stream and leave it on a
#    QUIET log for >2 minutes. If it dies around the 100s mark, the keepalive
#    isn't reaching the edge: either the backend isn't emitting it or NPM is
#    still buffering.
```

Test 3 is the one that matters. Tests 1 and 2 pass on a misconfigured stack.

---

## 5. If you skip the login (don't, but if you do)

If the decision is ever "Cloudflare mTLS is the only gate", then **before**
setting `HELM_DISABLE_AUTH=true`, change the web port binding in
`docker-compose.production.yml`:

```yaml
ports:
  - "127.0.0.1:5080:80"     # NOT "5080:80"
```

and put NPM on `helm-net` so it can still reach `helm-web` by service name.
Otherwise HELM is an unauthenticated control plane — able to unload the model
six agents depend on and delete 20GB of weights — listening on the LAN.
