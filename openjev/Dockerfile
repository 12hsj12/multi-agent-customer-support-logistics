# Run SiliconLabAI/OpenJev as a separate Node service. Pinning the source
# commit makes the deployed API implementation reproducible.
FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates

ARG OPENJEV_GIT_REF=a08e969c37b2e4a37f95b3426f983bd94303590c
WORKDIR /app
RUN git init /app && \
    git -C /app remote add origin https://github.com/SiliconLabAI/OpenJev.git && \
    git -C /app fetch --depth=1 origin "${OPENJEV_GIT_REF}" && \
    git -C /app checkout --detach FETCH_HEAD && \
    npm ci && npm run build

ENV NODE_ENV=production PORT=3001
EXPOSE 3001
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s --retries=5 \
    CMD node -e "fetch('http://127.0.0.1:3001/api/health').then(r => process.exit(r.ok ? 0 : 1)).catch(() => process.exit(1))"

CMD ["npm", "start"]
