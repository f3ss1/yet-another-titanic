import mlflow


class MLflowLoggingCallback:
    @staticmethod
    def after_iteration(info):
        for metric, value in info.metrics["learn"].items():
            mlflow.log_metric(metric, value[-1], step=info.iteration)
        return True
