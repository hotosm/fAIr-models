locals {
  domain_parts    = split(".", var.domain)
  domain_zone     = join(".", slice(local.domain_parts, length(local.domain_parts) - 2, length(local.domain_parts)))
  domain_prefix   = join(".", slice(local.domain_parts, 0, length(local.domain_parts) - 2))
  spaces_endpoint = "https://${var.region}.digitaloceanspaces.com"
}

# ---------- Kubernetes cluster ----------

data "digitalocean_kubernetes_versions" "this" {}

resource "digitalocean_kubernetes_cluster" "this" {
  name    = var.cluster_name
  region  = var.region
  version = data.digitalocean_kubernetes_versions.this.latest_version

  node_pool {
    name       = "infra"
    size       = var.infra_node_size
    node_count = 1
    labels = {
      "${var.domain}/role" = "infra"
    }
  }
}

resource "digitalocean_kubernetes_node_pool" "ml" {
  cluster_id = digitalocean_kubernetes_cluster.this.id
  name       = "ml"
  size       = var.ml_node_size
  auto_scale = true
  min_nodes  = 0
  max_nodes  = 1
  labels = {
    "${var.domain}/training"  = "true"
    "${var.domain}/inference" = "true"
  }
}

# ---------- Managed PostgreSQL ----------

resource "digitalocean_database_cluster" "this" {
  name       = "${var.cluster_name}-pg"
  engine     = "pg"
  version    = "17"
  region     = var.region
  size       = "db-s-2vcpu-4gb"
  node_count = 1
}

resource "digitalocean_database_db" "zenml" {
  cluster_id = digitalocean_database_cluster.this.id
  name       = "zenml"
}

resource "digitalocean_database_db" "fair_models" {
  cluster_id = digitalocean_database_cluster.this.id
  name       = "fair_models"
}

resource "digitalocean_database_db" "mlflow" {
  cluster_id = digitalocean_database_cluster.this.id
  name       = "mlflow"
}

resource "digitalocean_database_db" "mlflow_auth" {
  cluster_id = digitalocean_database_cluster.this.id
  name       = "mlflow_auth"
}

data "http" "local_ip" {
  url = "https://api.ipify.org"
}

resource "digitalocean_database_firewall" "this" {
  cluster_id = digitalocean_database_cluster.this.id

  rule {
    type  = "k8s"
    value = digitalocean_kubernetes_cluster.this.id
  }

  rule {
    type  = "ip_addr"
    value = trimspace(data.http.local_ip.response_body)
  }
}

# ---------- DO Spaces ----------

resource "digitalocean_spaces_bucket" "this" {
  name          = var.spaces_bucket
  region        = var.region
  acl           = "private"
  force_destroy = true
}

# ---------- Wildcard DNS ----------

resource "digitalocean_record" "wildcard" {
  domain = local.domain_zone
  type   = "A"
  name   = "*.${local.domain_prefix}"
  value  = data.kubernetes_service_v1.ingress_lb.status[0].load_balancer[0].ingress[0].ip
  ttl    = 300
}
