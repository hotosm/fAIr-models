output "zenml_url" {
  value = "https://zenml.${var.domain}"
}

output "mlflow_url" {
  value = "https://mlflow.${var.domain}"
}

output "stac_url" {
  value = "https://stac.${var.domain}/stac"
}

output "fair_domain" {
  value = var.domain
}

output "pgstac_dsn" {
  value     = "postgresql://${digitalocean_database_cluster.this.user}:${urlencode(digitalocean_database_cluster.this.password)}@${digitalocean_database_cluster.this.host}:${digitalocean_database_cluster.this.port}/fair_models?sslmode=require"
  sensitive = true
}

output "pg_uri" {
  value     = digitalocean_database_cluster.this.uri
  sensitive = true
}

output "spaces_endpoint" {
  value = local.spaces_endpoint
}

output "spaces_bucket" {
  value = var.spaces_bucket
}

output "doks_context" {
  value = "do-${var.region}-${var.cluster_name}"
}

output "spaces_access_key" {
  value     = var.spaces_access_key
  sensitive = true
}

output "spaces_secret_key" {
  value     = var.spaces_secret_key
  sensitive = true
}

output "mlflow_admin_user" {
  value = var.mlflow_admin_user
}

output "mlflow_admin_password" {
  value     = var.mlflow_admin_password
  sensitive = true
}

output "letsencrypt_email" {
  value = var.letsencrypt_email
}

output "zenml_admin_user" {
  value = var.zenml_admin_user
}

output "zenml_admin_password" {
  value     = var.zenml_admin_password
  sensitive = true
}
