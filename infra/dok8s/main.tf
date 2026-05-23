terraform {
  required_version = ">= 1.6"
  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.46"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.31"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "digitalocean" {
  token = var.do_token
}

provider "kubernetes" {
  host                   = digitalocean_kubernetes_cluster.this.endpoint
  cluster_ca_certificate = base64decode(digitalocean_kubernetes_cluster.this.kube_config[0].cluster_ca_certificate)
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "doctl"
    args        = ["kubernetes", "cluster", "kubeconfig", "exec-credential", "--version=v1beta1", digitalocean_kubernetes_cluster.this.id]
  }
}

data "digitalocean_kubernetes_versions" "this" {}

resource "digitalocean_kubernetes_cluster" "this" {
  name    = var.cluster_name
  region  = var.region
  version = data.digitalocean_kubernetes_versions.this.latest_version

  node_pool {
    name       = "infra"
    size       = "s-2vcpu-4gb"
    node_count = 1
  }

  lifecycle {
    ignore_changes = [node_pool]
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

resource "digitalocean_kubernetes_node_pool" "system" {
  cluster_id = digitalocean_kubernetes_cluster.this.id
  name       = "infra-v2"
  size       = var.system_node_size
  node_count = 1
  labels = {
    "fair/role" = "infra"
  }
}

resource "digitalocean_kubernetes_node_pool" "gpu" {
  cluster_id = digitalocean_kubernetes_cluster.this.id
  name       = "gpu"
  size       = var.gpu_node_size
  auto_scale = true
  min_nodes  = 0
  max_nodes  = var.gpu_max_nodes
  labels = {
    "fair/workload"           = "ml-gpu"
    "${var.domain}/training"  = "true"
    "${var.domain}/inference" = "true"
  }
}

