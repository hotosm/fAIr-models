from __future__ import annotations

from unittest.mock import MagicMock, patch


def _mock_mlflow():
    mock = MagicMock()
    mock.autolog = MagicMock()
    mock.log_params = MagicMock()
    mock.log_metrics = MagicMock()
    mock.set_tags = MagicMock()
    return mock


class TestMlflowTrainingContext:
    def test_logs_params_and_wall_time(self) -> None:
        mock_mlflow = _mock_mlflow()
        with (
            patch.dict("sys.modules", {"mlflow": mock_mlflow}),
            patch("fair.zenml.instrumentation.log_training_wall_time") as mock_wall,
        ):
            from fair.zenml.instrumentation import mlflow_training_context

            hyperparams = {"epochs": 10, "learning_rate": 0.001}
            with mlflow_training_context(hyperparams, model_name="test-model"):
                pass

            mock_mlflow.autolog.assert_called_once()
            mock_mlflow.log_params.assert_called_once_with(hyperparams)
            mock_mlflow.set_tags.assert_called_once()
            tags = mock_mlflow.set_tags.call_args[0][0]
            assert tags["fair.model_name"] == "test-model"
            mock_wall.assert_called_once()
            assert mock_wall.call_args[0][0] > 0

    def test_filters_non_scalar_params(self) -> None:
        mock_mlflow = _mock_mlflow()
        with (
            patch.dict("sys.modules", {"mlflow": mock_mlflow}),
            patch("fair.zenml.instrumentation.log_training_wall_time"),
        ):
            from fair.zenml.instrumentation import mlflow_training_context

            hyperparams = {"epochs": 5, "nested": {"a": 1}, "tags": ["x"]}
            with mlflow_training_context(hyperparams):
                pass

            logged = mock_mlflow.log_params.call_args[0][0]
            assert "epochs" in logged
            assert "nested" not in logged
            assert "tags" not in logged

    def test_sets_all_tags(self) -> None:
        mock_mlflow = _mock_mlflow()
        with (
            patch.dict("sys.modules", {"mlflow": mock_mlflow}),
            patch("fair.zenml.instrumentation.log_training_wall_time"),
        ):
            from fair.zenml.instrumentation import mlflow_training_context

            with mlflow_training_context({}, model_name="m", base_model_id="b", dataset_id="d"):
                pass

            tags = mock_mlflow.set_tags.call_args[0][0]
            assert tags == {
                "fair.model_name": "m",
                "fair.base_model": "b",
                "fair.dataset": "d",
            }

    def test_no_tags_when_none_provided(self) -> None:
        mock_mlflow = _mock_mlflow()
        with (
            patch.dict("sys.modules", {"mlflow": mock_mlflow}),
            patch("fair.zenml.instrumentation.log_training_wall_time"),
        ):
            from fair.zenml.instrumentation import mlflow_training_context

            with mlflow_training_context({}):
                pass

            mock_mlflow.set_tags.assert_not_called()


class TestLogEvaluationResults:
    def test_logs_to_both_backends(self) -> None:
        mock_mlflow = _mock_mlflow()
        with (
            patch.dict("sys.modules", {"mlflow": mock_mlflow}),
            patch("fair.zenml.instrumentation.log_fair_metrics") as mock_fair,
        ):
            from fair.zenml.instrumentation import log_evaluation_results

            metrics = {"accuracy": 0.95, "loss": 0.05, "notes": "good"}
            log_evaluation_results(metrics)

            mock_mlflow.log_metrics.assert_called_once_with({"accuracy": 0.95, "loss": 0.05})
            mock_fair.assert_called_once_with(metrics)
