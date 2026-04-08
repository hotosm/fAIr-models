-- Initialize databases for the fAIr dev stack.
-- Runs once via PG initdb (mounted as /docker-entrypoint-initdb.d/).

CREATE DATABASE zenml;
CREATE DATABASE fair_models;
CREATE DATABASE mlflow;

\c fair_models
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS btree_gist;
