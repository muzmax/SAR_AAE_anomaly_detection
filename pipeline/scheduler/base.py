import abc


class SchedulerBase(abc.ABC):
    @abc.abstractmethod
    def step(self, loss, metrics, epoch_id):
        pass
    @abc.abstractmethod
    def load(self, param):
        pass
    @abc.abstractmethod
    def get_param(self):
        pass


class SchedulerWrapperBase(SchedulerBase):
    def __init__(self, scheduler):
        self._scheduler = scheduler


class SchedulerWrapperIdentity(SchedulerWrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(None)

    def step(self, loss, metrics, epoch_id):
        pass
    def load(self,param):
        pass
    def get_param(self):
        return {}


class SchedulerWrapperLossBase(SchedulerWrapperBase):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def step(self, loss, metrics, epoch_id):
        return self._scheduler.step(loss, epoch_id)


class SchedulerWrapperMetricsMeanBase(SchedulerWrapperBase):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def step(self, loss, metrics, epoch_id):
        values = list(metrics.values())
        mean_metrics = sum(values) / len(values)
        return self._scheduler.step(mean_metrics, epoch_id)
