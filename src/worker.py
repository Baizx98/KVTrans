from cache_engine import CacheEngine
from config import CacheConfig, ModelConfig, DeviceConfig
from time import sleep


class Worker:
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_engine = CacheEngine(
            cache_config,
            model_config,
            device_config,
        )

    def execute_model(self, input_data=None):
        print("Executing model with input")
        sleep(0.2)  # Simulate model execution time
        print("Model execution completed.")
