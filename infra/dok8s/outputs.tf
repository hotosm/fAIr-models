output "cluster_id" {
  value = digitalocean_kubernetes_cluster.this.id
}

output "doks_context" {
  value = "do-${var.region}-${var.cluster_name}"
}

output "fair_domain" {
  value = var.domain
}

output "letsencrypt_email" {
  value = var.letsencrypt_email
}

output "mlflow_admin_user" {
  value = var.mlflow_admin_user
}

output "mlflow_admin_password" {
  value     = var.mlflow_admin_password
  sensitive = true
}

output "zenml_admin_user" {
  value = var.zenml_admin_user
}

output "zenml_admin_password" {
  value     = var.zenml_admin_password
  sensitive = true
}

output "zenml_store_api_key" {
  value     = var.zenml_store_api_key
  sensitive = true
}
