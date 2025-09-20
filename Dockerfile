# Stage 1: build frontend
FROM node:18 AS frontend-build
WORKDIR /app/frontend

# package.json is in Scripts/
COPY Scripts/package.json Scripts/package-lock.json* ./
COPY package.json package-lock.json* ./
# copy index.html (Vite expects it at project root)
COPY Scripts/index.html ./
# frontend sources live in Scripts/src
COPY Scripts/src ./src
COPY Scripts/public ./public
COPY Scripts/vite.config.ts ./

# Use npm ci when lockfile exists, otherwise fall back to npm install
#RUN bash -lc 'if [ -f package-lock.json ]; then npm ci --silent; else npm install --silent; fi'
RUN npm install --legacy-peer-deps
RUN npm run build

# Stage 2: build backend image
FROM python:3.11-slim
WORKDIR /app

# system deps needed for psycopg2 / building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# install python deps
COPY Scripts/requirements.txt ./requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
# ensure gunicorn is available even if not in requirements
RUN pip install --no-cache-dir gunicorn

# copy backend source
COPY Scripts/py-src/data_formulator ./data_formulator


# copy built frontend into Flask static directory
RUN mkdir -p /app/data_formulator/static
COPY --from=frontend-build /app/frontend/py-src/data_formulator/dist/ /app/data_formulator/static/


ENV PORT=8080
EXPOSE 8080

# run with gunicorn; adjust module path if your Flask app entry differs
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "data_formulator.app:app", "--workers", "2", "--timeout", "120"]