# syntax=docker/dockerfile:1.7

ARG KUBECTL_VERSION=v1.32.2
ARG KIND_VERSION=v0.27.0
ARG HELM_VERSION=v3.18.4
ARG HELM_DIFF_VERSION=3.12.5
ARG HELMFILE_VERSION=0.171.0
ARG JUST_VERSION=1.40.0
ARG MC_VERSION=RELEASE.2025-08-13T08-35-41Z

FROM debian:trixie-slim AS builder
ARG KUBECTL_VERSION
ARG KIND_VERSION
ARG HELM_VERSION
ARG HELM_DIFF_VERSION
ARG HELMFILE_VERSION
ARG JUST_VERSION
ARG MC_VERSION
ARG TARGETARCH=amd64

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates tar git upx-ucl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /out/bin /out/helm-data

RUN curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${TARGETARCH}/kubectl" \
        -o /out/bin/kubectl && chmod +x /out/bin/kubectl

RUN curl -fsSL "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${TARGETARCH}" \
        -o /out/bin/kind && chmod +x /out/bin/kind

RUN curl -fsSL "https://get.helm.sh/helm-${HELM_VERSION}-linux-${TARGETARCH}.tar.gz" \
        | tar -xzC /tmp \
    && mv "/tmp/linux-${TARGETARCH}/helm" /out/bin/helm \
    && rm -rf "/tmp/linux-${TARGETARCH}"

RUN curl -fsSL "https://github.com/helmfile/helmfile/releases/download/v${HELMFILE_VERSION}/helmfile_${HELMFILE_VERSION}_linux_${TARGETARCH}.tar.gz" \
        | tar -xzC /out/bin helmfile

RUN curl -fsSL "https://dl.min.io/client/mc/release/linux-${TARGETARCH}/archive/mc.${MC_VERSION}" \
        -o /out/bin/mc && chmod +x /out/bin/mc

RUN curl -fsSL "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
        | tar -xzC /out/bin just

ENV HELM_DATA_HOME=/out/helm-data
RUN /out/bin/helm plugin install https://github.com/databus23/helm-diff \
        --version "${HELM_DIFF_VERSION}" \
    && cd /out/helm-data/plugins/helm-diff \
    && find . -mindepth 1 -maxdepth 1 \
         ! -name plugin.yaml ! -name bin ! -name install-binary.sh \
         -exec rm -rf {} +

RUN upx --best --lzma \
        /out/bin/kubectl \
        /out/bin/kind \
        /out/bin/helm \
        /out/bin/helmfile \
        /out/bin/mc \
        /out/bin/just \
        /out/helm-data/plugins/helm-diff/bin/diff


FROM alpine:3.21

RUN apk add --no-cache bash docker-cli ca-certificates netcat-openbsd

COPY --from=builder /out/bin/kubectl  /usr/local/bin/kubectl
COPY --from=builder /out/bin/kind     /usr/local/bin/kind
COPY --from=builder /out/bin/helm     /usr/local/bin/helm
COPY --from=builder /out/bin/helmfile /usr/local/bin/helmfile
COPY --from=builder /out/bin/mc       /usr/local/bin/mc
COPY --from=builder /out/bin/just     /usr/local/bin/just
COPY --from=builder /out/helm-data    /root/.local/share/helm

ENV PATH=/usr/local/bin:$PATH \
    HOME=/root \
    HELM_DATA_HOME=/root/.local/share/helm

WORKDIR /workspace
CMD ["bash"]
