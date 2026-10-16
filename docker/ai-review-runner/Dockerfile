# Образ ephemeral self-hosted runner для AI-ревью PR (README, «AI-ревью pull request»).
# Зависимости ревью ставятся ЗДЕСЬ, а не в job: код PR ничего не устанавливает и не запускает.
FROM ubuntu:24.04

ARG RUNNER_VERSION=2.338.0

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git gh jq python3 python3-pip libicu74 \
 && pip install --no-cache-dir --break-system-packages openai httpx faiss-cpu numpy \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m runner \
 && mkdir /index && chown runner:runner /index   # том индекса наследует владельца

USER runner
WORKDIR /home/runner

RUN curl -fsSL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
    | tar xz

COPY --chown=runner:runner entrypoint.sh /home/runner/entrypoint.sh
RUN chmod +x /home/runner/entrypoint.sh

ENTRYPOINT ["/home/runner/entrypoint.sh"]
