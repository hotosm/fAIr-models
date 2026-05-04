terraform {
  required_version = ">= 1.6"
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.46"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.5"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

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
      "fair/role" = "infra"
    }
  }
}

resource "digitalocean_kubernetes_node_pool" "ml" {
  cluster_id = digitalocean_kubernetes_cluster.this.id
  name       = "ml"
  size       = var.ml_node_size
  auto_scale = true
  min_nodes  = 0
  max_nodes  = var.ml_max_nodes
  labels = {
    "fair/workload"           = "ml"
    "${var.domain}/training"  = "true"
    "${var.domain}/inference" = "true"
  }
}

resource "local_file" "env_helmfile" {
  filename        = "${path.module}/.env.helmfile"
  file_permission = "0600"
  content         = <<-EOT
    export FAIR_DOMAIN=${var.domain}
    export LETSENCRYPT_EMAIL=${var.letsencrypt_email}
    export MLFLOW_ADMIN_USER=${var.mlflow_admin_user}
    export MLFLOW_ADMIN_PASSWORD=${var.mlflow_admin_password}
    export ZENML_ADMIN_USER=${var.zenml_admin_user}
    export ZENML_ADMIN_PASSWORD=${var.zenml_admin_password}
  EOT
}
