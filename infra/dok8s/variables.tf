variable "do_token" {
  type      = string
  sensitive = true
}

variable "cluster_name" {
  type    = string
  default = "fair"
}

variable "region" {
  type    = string
  default = "nyc3"
}

variable "domain" {
  type = string
}

variable "infra_node_size" {
  type    = string
  default = "s-2vcpu-4gb"
}

variable "ml_node_size" {
  type    = string
  default = "s-4vcpu-8gb"
}

variable "spaces_bucket" {
  type = string
}

variable "spaces_access_key" {
  type      = string
  sensitive = true
}

variable "spaces_secret_key" {
  type      = string
  sensitive = true
}

variable "mlflow_admin_user" {
  type = string
}

variable "mlflow_admin_password" {
  type      = string
  sensitive = true
}

variable "zenml_admin_user" {
  type = string
}

variable "zenml_admin_password" {
  type      = string
  sensitive = true
}

variable "letsencrypt_email" {
  type = string
}
