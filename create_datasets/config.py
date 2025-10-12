"""
Конфигурационный файл для генерации датасета
"""

import numpy as np
import os

class DatasetConfig:
    """Конфигурация для генерации датасета"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        # Основные параметры датасета
        self.num_trajectories = 2000  # Количество траекторий для генерации
        self.num_jobs = 5  # Количество процессов для параллельной обработки

        # Параметры интегрирования
        self.t_end = 5000  # Время интегрирования
        self.dt = 0.01  # Шаг интегрирования
        self.transient_time = 100  # Время переходного процесса (исключается из результата)

        # Начальные условия
        self.initial_conditions = [1.0, 0.0]  # [x(0), x'(0)]

        # Директория для сохранения датасета
        self.output_dir = "datasets"
        self.dataset_name = "mathieu_delayed_dataset"

        # Параметры, которые варьируются (равномерное распределение)
        # Диапазоны основаны на примере test6 с разумными вариациями
        self.variable_params = {
            'b': (0.3, 0.38),
            'b_e': (0.36, 0.44),
            'nonlin_k': (0.03, 0.05),   # нелинейный коэффициент демпфирования
            'F_As': (3.0, 5.0),
            'F_Ae': (3.0, 5.0),
            'b_ts': (0, 3000),         # время начала изменения коэффициента запаздывания
            'b_te': (1000, 4000),         # время окончания изменения коэффициента запаздывания
        }

        # Константные параметры
        self.constant_params = {
            'k': 0.181,           # коэффициент демпфирования
            'd': 2.5,      # основная частота
            'e': 0,      # амплитуда модуляции
            'tau': 2*np.pi,  # время запаздывания
            'T': 2*np.pi,         # период модуляции
            'F_ts': 0,            # время начала изменения внешней силы
            'F_te': 10,           # время окончания изменения внешней силы
            'F_T': 2*np.pi/3,     # период внешней силы
        }

        # Параметры для фильтрации траекторий
        self.filter_config = {
            'max_amplitude': 100.0,  # максимальная допустимая амплитуда
            'min_amplitude': 0.01,   # минимальная допустимая амплитуда
            'check_stability': True,  # проверять стабильность решения
        }

        # Параметры сохранения
        self.save_config = {
            'save_plots': False,     # сохранять ли графики траекторий
            'save_metadata': True,   # сохранять ли метаданные
            'compression': True,     # использовать ли сжатие при сохранении
        }

    def get_output_path(self):
        """Возвращает полный путь для сохранения датасета"""
        return os.path.join(self.output_dir, f"{self.dataset_name}.pkl")

    def get_metadata_path(self):
        """Возвращает путь для сохранения метаданных"""
        return os.path.join(self.output_dir, f"{self.dataset_name}_metadata.pkl")

    def create_output_dir(self):
        """Создает директорию для вывода если она не существует"""
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_random_params(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        """Генерирует случайный набор параметров"""
        params = self.constant_params.copy()

        # Добавляем случайные значения для варьируемых параметров
        for param_name, (min_val, max_val) in self.variable_params.items():
            params[param_name] = np.random.uniform(min_val, max_val)

        if params['b_ts'] >= params['b_te']:
            params['b_ts'], params['b_te'] = params['b_te'], params['b_ts']
        if params["b_te"] - params["b_ts"] < 100:
            params["b_te"] = params["b_ts"] + 100
        params["F_As"] = params["F_Ae"]

        return params

    def validate_config(self):
        """Проверяет корректность конфигурации"""
        errors = []

        if self.num_trajectories <= 0:
            errors.append("num_trajectories должно быть положительным")

        if self.num_jobs <= 0:
            errors.append("num_jobs должно быть положительным")

        if self.t_end <= 0:
            errors.append("t_end должно быть положительным")

        if self.dt <= 0:
            errors.append("dt должно быть положительным")

        # Проверяем диапазоны параметров
        for param_name, (min_val, max_val) in self.variable_params.items():
            if min_val >= max_val:
                errors.append(f"Неверный диапазон для {param_name}: min >= max")

        if errors:
            raise ValueError("Ошибки в конфигурации:\n" + "\n".join(errors))

        return True

    def print_config(self):
        """Выводит текущую конфигурацию"""
        print("=" * 60)
        print("КОНФИГУРАЦИЯ ГЕНЕРАЦИИ ДАТАСЕТА")
        print("=" * 60)
        print(f"Количество траекторий: {self.num_trajectories}")
        print(f"Количество процессов: {self.num_jobs}")
        print(f"Время интегрирования: {self.t_end}")
        print(f"Шаг интегрирования: {self.dt}")
        print(f"Переходный процесс: {self.transient_time}")
        print(f"Директория вывода: {self.output_dir}")
        print(f"Имя датасета: {self.dataset_name}")
        print()
        print("Варьируемые параметры:")
        for param, (min_val, max_val) in self.variable_params.items():
            print(f"  {param}: [{min_val:.3f}, {max_val:.3f}]")
        print()
        print("Константные параметры:")
        for param, value in self.constant_params.items():
            print(f"  {param}: {value}")
        print("=" * 60)


# Создаем экземпляр конфигурации по умолчанию
default_config = DatasetConfig()
