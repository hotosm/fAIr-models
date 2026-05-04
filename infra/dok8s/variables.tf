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
  type        = string
  description = "Wildcard base domain, e.g. fair.example.com"
}

variable "infra_node_size" {
  type    = string
  default = "s-2vcpu-4gb"
}

variable "ml_node_size" {
  type    = string
  default = "s-4vcpu-8gb"
}

variable "ml_max_nodes" {
  type    = number
  default = 5
}

variable "letsencrypt_email" {
  type = string
}

variable "mlflow_admin_user" {
  type    = string
  default = "admin"
}

variable "mlflow_admin_password" {
  type      = string
  sensitive = true
}

variable "zenml_admin_user" {
  type    = string
  default = "default"
}

variable "zenml_admin_password" {
  type      = string
  sensitive = true
}
