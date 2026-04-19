resource "kubernetes_namespace" "fair" {
  metadata { name = "fair" }
}

# ---------- Ingress tier ----------

resource "helm_release" "cert_manager" {
  name             = "cert-manager"
  namespace        = "cert-manager"
  create_namespace = true
  repository       = "https://charts.jetstack.io"
  chart            = "cert-manager"
  version          = "v1.20.1"
  timeout          = 300
  wait             = true

  set {
    name  = "crds.enabled"
    value = "true"
  }
}

resource "helm_release" "ingress_nginx" {
  name             = "ingress-nginx"
  namespace        = "ingress-nginx"
  create_namespace = true
  repository       = "https://kubernetes.github.io/ingress-nginx"
  chart            = "ingress-nginx"
  version          = "4.15.1"
  timeout          = 300
  wait             = true

  values = [yamlencode({
    controller = {
      service = {
        annotations = {
          "service.beta.kubernetes.io/do-loadbalancer-name"      = "fair-lb"
          "service.beta.kubernetes.io/do-loadbalancer-size-unit" = "1"
        }
      }
      resources = {
        requests = { cpu = "100m", memory = "128Mi" }
        limits   = { cpu = "500m", memory = "256Mi" }
      }
    }
  })]

  depends_on = [helm_release.cert_manager]
}

data "kubernetes_service_v1" "ingress_lb" {
  metadata {
    name      = "ingress-nginx-controller"
    namespace = "ingress-nginx"
  }
  depends_on = [helm_release.ingress_nginx]
}

# ---------- App tier ----------

resource "helm_release" "stac" {
  name       = "stac"
  namespace  = kubernetes_namespace.fair.metadata[0].name
  repository = "https://devseed.com/eoapi-k8s/"
  chart      = "eoapi"
  version    = "0.12.2"
  timeout    = 600
  wait       = true

  values = [yamlencode({
    postgresql = {
      type = "external-plaintext"
      external = {
        host     = digitalocean_database_cluster.this.host
        port     = tostring(digitalocean_database_cluster.this.port)
        database = "fair_models"
        credentials = {
          username = digitalocean_database_cluster.this.user
          password = digitalocean_database_cluster.this.password
        }
        sslmode = "require"
      }
    }
    postgrescluster = { enabled = false }
    pgstacBootstrap = { enabled = false }
    stac = {
      resources = {
        requests = { cpu = "100m", memory = "256Mi" }
        limits   = { cpu = "500m", memory = "512Mi" }
      }
      ingress          = { enabled = false }
      overrideRootPath = ""
      settings = {
        envVars = {
          HOST                           = "0.0.0.0"
          PORT                           = "8080"
          WEB_CONCURRENCY                = "2"
          DB_MIN_CONN_SIZE               = "1"
          DB_MAX_CONN_SIZE               = "5"
          PGSSLMODE                      = "require"
          STAC_FASTAPI_TITLE             = "fAIr STAC Catalog"
          STAC_FASTAPI_DESCRIPTION       = "fAIr model metadata catalog on DigitalOcean."
          ENABLE_TRANSACTIONS_EXTENSIONS = "true"
        }
      }
    }
    browser           = { enabled = false }
    docServer         = { enabled = false }
    raster            = { enabled = false }
    multidim          = { enabled = false }
    "stac-auth-proxy" = { enabled = false }
    vector            = { enabled = false }
    "eoapi-notifier"  = { enabled = false }
    monitoring        = { metricsServer = { enabled = false }, prometheus = { enabled = false } }
    observability     = { grafana = { enabled = false } }
    ingress           = { enabled = false }
  })]

  depends_on = [helm_release.ingress_nginx, digitalocean_database_db.fair_models]
}

resource "kubernetes_ingress_v1" "stac" {
  metadata {
    name      = "stac"
    namespace = kubernetes_namespace.fair.metadata[0].name
    annotations = {
      "cert-manager.io/cluster-issuer"             = "letsencrypt"
      "nginx.ingress.kubernetes.io/use-regex"      = "true"
      "nginx.ingress.kubernetes.io/rewrite-target" = "/$2"
    }
  }

  spec {
    ingress_class_name = "nginx"

    tls {
      secret_name = "stac-tls"
      hosts       = ["stac.${var.domain}"]
    }

    rule {
      host = "stac.${var.domain}"
      http {
        path {
          path      = "/(/|$)(.*)"
          path_type = "ImplementationSpecific"
          backend {
            service {
              name = "stac-stac"
              port { number = 8080 }
            }
          }
        }
        path {
          path      = "/stac(/|$)(.*)"
          path_type = "ImplementationSpecific"
          backend {
            service {
              name = "stac-stac"
              port { number = 8080 }
            }
          }
        }
      }
    }
  }

  depends_on = [helm_release.stac]
}

resource "helm_release" "mlflow" {
  name       = "mlflow"
  namespace  = kubernetes_namespace.fair.metadata[0].name
  repository = "https://community-charts.github.io/helm-charts"
  chart      = "mlflow"
  version    = "1.8.1"
  timeout    = 300
  wait       = true

  values = [yamlencode({
    backendStore = {
      databaseMigration = true
      postgres = {
        enabled  = true
        host     = digitalocean_database_cluster.this.host
        port     = digitalocean_database_cluster.this.port
        database = "mlflow"
        user     = digitalocean_database_cluster.this.user
        password = digitalocean_database_cluster.this.password
      }
    }
    artifactRoot = {
      s3 = {
        enabled            = true
        bucket             = var.spaces_bucket
        path               = "mlflow"
        awsAccessKeyId     = var.spaces_access_key
        awsSecretAccessKey = var.spaces_secret_key
      }
    }
    auth = {
      enabled       = true
      adminUsername = var.mlflow_admin_user
      adminPassword = var.mlflow_admin_password
      postgres = {
        enabled  = true
        host     = digitalocean_database_cluster.this.host
        port     = digitalocean_database_cluster.this.port
        database = "mlflow_auth"
        user     = digitalocean_database_cluster.this.user
        password = digitalocean_database_cluster.this.password
        driver   = "psycopg2"
      }
    }
    ingress = {
      enabled     = true
      className   = "nginx"
      annotations = { "cert-manager.io/cluster-issuer" = "letsencrypt" }
      hosts = [{
        host  = "mlflow.${var.domain}"
        paths = [{ path = "/", pathType = "Prefix" }]
      }]
      tls = [{
        secretName = "mlflow-tls"
        hosts      = ["mlflow.${var.domain}"]
      }]
    }
    resources = {
      requests = { cpu = "100m", memory = "512Mi" }
      limits   = { cpu = "500m", memory = "1Gi" }
    }
    extraEnvVars = {
      PGSSLMODE                   = "require"
      MLFLOW_S3_ENDPOINT_URL      = local.spaces_endpoint
      MLFLOW_SERVER_ALLOWED_HOSTS = "*"
    }
  })]

  depends_on = [helm_release.ingress_nginx, digitalocean_database_db.mlflow, digitalocean_database_db.mlflow_auth]
}

# ---------- KNative live serving tier ----------

resource "kubernetes_namespace" "predict" {
  metadata { name = "predict" }
}

resource "helm_release" "knative_operator" {
  name             = "knative-operator"
  namespace        = "knative-operator"
  create_namespace = true
  repository       = "https://knative.github.io/operator"
  chart            = "knative-operator"
  version          = "v1.17.0"
  timeout          = 600
  wait             = true
}

resource "kubernetes_manifest" "knative_serving" {
  manifest = {
    apiVersion = "operator.knative.dev/v1beta1"
    kind       = "KnativeServing"
    metadata = {
      name      = "knative-serving"
      namespace = "knative-serving"
    }
    spec = {
      ingress = {
        kourier = { enabled = true }
      }
      config = {
        network = {
          "ingress-class" = "kourier.ingress.networking.knative.dev"
        }
        autoscaler = {
          "enable-scale-to-zero"       = "true"
          "scale-to-zero-grace-period" = "30s"
        }
      }
    }
  }

  depends_on = [helm_release.knative_operator]
}

resource "kubernetes_ingress_v1" "predict" {
  metadata {
    name      = "predict"
    namespace = "knative-serving"
    annotations = {
      "cert-manager.io/cluster-issuer" = "letsencrypt"
    }
  }

  spec {
    ingress_class_name = "nginx"

    tls {
      secret_name = "predict-tls"
      hosts       = ["predict.${var.domain}"]
    }

    rule {
      host = "predict.${var.domain}"
      http {
        path {
          path      = "/"
          path_type = "Prefix"
          backend {
            service {
              name = "kourier"
              port { number = 80 }
            }
          }
        }
      }
    }
  }

  depends_on = [kubernetes_manifest.knative_serving]
}

resource "helm_release" "zenml" {
  name       = "zenml"
  namespace  = kubernetes_namespace.fair.metadata[0].name
  repository = "oci://public.ecr.aws/zenml"
  chart      = "zenml"
  version    = "0.93.3"
  timeout    = 480
  wait       = true

  values = [yamlencode({
    zenml = {
      image = {
        repository = "ghcr.io/hotosm/zenml-postgres"
        tag        = "0.93.3"
        pullPolicy = "IfNotPresent"
      }
      debug          = false
      analyticsOptIn = false
      ingress = {
        enabled          = true
        ingressClassName = "nginx"
        annotations      = { "cert-manager.io/cluster-issuer" = "letsencrypt" }
        host             = "zenml.${var.domain}"
        tls = {
          enabled       = true
          generateCerts = false
        }
      }
      database = {
        url            = "postgresql://${digitalocean_database_cluster.this.user}:${digitalocean_database_cluster.this.password}@${digitalocean_database_cluster.this.host}:${digitalocean_database_cluster.this.port}/zenml?sslmode=require"
        backupStrategy = "disabled"
      }
      environment = {
        ZENML_SERVER_AUTO_ACTIVATE   = "1"
        ZENML_DEFAULT_USER_NAME      = var.zenml_admin_user
        ZENML_DEFAULT_USER_PASSWORD  = var.zenml_admin_password
        ZENML_STORE_SQL_POOL_SIZE    = "10"
        ZENML_STORE_SQL_MAX_OVERFLOW = "5"
      }
      resources = {
        requests = { cpu = "250m", memory = "512Mi" }
        limits   = { cpu = "1", memory = "1Gi" }
      }
    }
  })]

  depends_on = [helm_release.ingress_nginx, digitalocean_database_db.zenml]
}
