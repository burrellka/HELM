# HELM web. Build from the REPO ROOT:
#   docker build -f frontend/Dockerfile -t brain40/helm-web:0.1.0 .
FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Carries the resolver trick + proxy_buffering off. Read it before editing.
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
