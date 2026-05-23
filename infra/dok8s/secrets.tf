# Cluster-side secrets. Pods consume these via `envFrom: secretRef` in their
# Deployments — see manifests/fair-backend.yaml. Add a new key here, run
# `tofu apply`, restart the consuming Deployment.

resource "random_password" "fair_secret_key" {
  length  = 64
  special = false
}

resource "random_password" "fair_dev_token" {
  length  = 48
  special = false
}

resource "kubernetes_secret" "fair_backend" {
  metadata {
    name      = "fair-backend-secrets"
    namespace = "fair"
  }

  type = "Opaque"

  data = {
    SECRET_KEY               = random_password.fair_secret_key.result
    DATABASE_URL             = "postgis://postgres:postgres@postgres.fair.svc.cluster.local:5432/fair"
    FAIR_DEV_TOKEN           = random_password.fair_dev_token.result
    FAIR_ZENML_STORE_API_KEY = var.zenml_store_api_key
    ZENML_STORE_API_KEY      = var.zenml_store_api_key
    AWS_ACCESS_KEY_ID        = "minioadmin"
    AWS_SECRET_ACCESS_KEY    = "minioadmin"
  }

  # Keep existing SECRET_KEY / FAIR_DEV_TOKEN values across imports so live
  # tokens don't rotate on every apply.
  lifecycle {
    ignore_changes = [
      data["SECRET_KEY"],
      data["FAIR_DEV_TOKEN"],
    ]
  }
}

# Pre-created so the digitalocean_dns Secret below can land before helmfile
# installs cert-manager. ClusterIssuer secret refs resolve to this namespace.
resource "kubernetes_namespace" "cert_manager" {
  metadata {
    name = "cert-manager"
  }
  lifecycle {
    ignore_changes = [metadata[0].labels, metadata[0].annotations]
  }
}

resource "kubernetes_secret" "digitalocean_dns" {
  metadata {
    name      = "digitalocean-dns"
    namespace = kubernetes_namespace.cert_manager.metadata[0].name
  }

  type = "Opaque"

  data = {
    access-token = var.do_token
  }
}
