"""
Генератор датасета решений уравнения Матье с запаздыванием

Этот пакет предоставляет инструменты для создания датасетов, состоящих из множества 
решений модифицированного уравнения Матье с запаздыванием с различными параметрами.

Основные компоненты:
- DelayedMathieuEquationJIT: Решатель уравнения
- DatasetGenerator: Генератор датасета
- DatasetConfig: Конфигурация параметров

Пример использования:
    from config import DatasetConfig
    from dataset_generator import DatasetGenerator
    
    config = DatasetConfig()
    config.num_trajectories = 100
    
    generator = DatasetGenerator(config)
    dataset = generator.generate_dataset()
    generator.save_dataset(dataset)
"""

__version__ = "1.0.0"
__author__ = "Dataset Generator"
__email__ = ""

# Импорты для удобства использования
from .equation_solver import DelayedMathieuEquationJIT
from .dataset_generator import DatasetGenerator
from .config import DatasetConfig, default_config

__all__ = [
    'DelayedMathieuEquationJIT',
    'DatasetGenerator', 
    'DatasetConfig',
    'default_config'
]