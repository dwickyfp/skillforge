# Multi-stage Dockerfile for SkillForge
# Stage 1: Build React frontend
# Stage 2: Python + nginx runtime

# ============================================
# Stage 1: Build Frontend
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/dashboard

# Copy package files
COPY dashboard/package.json dashboard/package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy dashboard source
COPY dashboard/ ./

# Build production bundle
RUN npm run build

# ============================================
# Stage 2: Python + nginx runtime
# ============================================
FROM python:3.11-slim

# Install nginx and supervisor
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Python backend
COPY skillforge/ ./skillforge/
COPY pyproject.toml ./

# Install Python package in development mode (stdlib-only, no pip deps)
# Actually, since it's stdlib-only, we can just add to PYTHONPATH
ENV PYTHONPATH=/app

# Copy built frontend from builder stage
COPY --from=frontend-builder /app/dashboard/dist /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/sites-available/default

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy startup script and Python server entry point
COPY docker/start.sh /start.sh
COPY docker/run_server.py /app/docker/run_server.py
RUN chmod +x /start.sh

# Create data directory for SQLite
RUN mkdir -p /app/data

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/api/v1/health || exit 1

# Start supervisor (manages both nginx and python)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
