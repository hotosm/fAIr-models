output "cluster_id" {
  value = digitalocean_kubernetes_cluster.this.id
}

output "doks_context" {
  value = "do-${var.region}-${var.cluster_name}"
}

output "fair_domain" {
  value = var.domain
}
