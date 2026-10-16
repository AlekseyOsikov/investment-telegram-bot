# Образ ephemeral self-hosted runner для AI-ревью PR (README, «AI-ревью pull request»).
# Зависимости ревью ставятся ЗДЕСЬ, а не в job: код PR ничего не устанавливает и не запускает.
FROM ubuntu:24.04

ARG RUNNER_VERSION=2.338.0
# SHA256 архива из описания релиза (github.com/actions/runner/releases): сверяется при сборке.
ARG RUNNER_SHA256=af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git gh jq python3 python3-pip libicu74 \
 && pip install --no-cache-dir --break-system-packages \
      openai==3.31.0 httpx==0.28.1 faiss-cpu==1.15.1 numpy==2.5.4 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd -m runner \
 && mkdir /index && chown runner:runner /index   # том индекса наследует владельца

USER runner
WORKDIR /home/runner

RUN curl -fsSL -o runner.tar.gz \
      "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz" \
 && echo "${RUNNER_SHA256}  runner.tar.gz" | sha256sum -c - \
 && tar xzf runner.tar.gz && rm runner.tar.gz

COPY --chown=runner:runner entrypoint.sh /home/runner/entrypoint.sh
RUN chmod +x /home/runner/entrypoint.sh

ENTRYPOINT ["/home/runner/entrypoint.sh"]
