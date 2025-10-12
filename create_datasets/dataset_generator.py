"""
Основной модуль для генерации датасета решений уравнения Матье с запаздыванием
"""

import numpy as np
import pickle
import time
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional
import traceback
import os
from tqdm import tqdm

from equation_solver import DelayedMathieuEquationJIT as DelayedMathieuEquationJIT
from config import DatasetConfig


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Генератор датасета решений дифференциального уравнения"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.config.validate_config()
        self.config.create_output_dir()
        
        # Статистика генерации
        self.stats = {
            'total_generated': 0,
            'successful': 0,
            'failed': 0,
            'filtered_out': 0,
            'generation_time': 0.0
        }
    
    def generate_single_trajectory(self, trajectory_id: int) -> Optional[Dict]:
        """
        Генерирует одну траекторию
        
        Args:
            trajectory_id: Уникальный идентификатор траектории
            
        Returns:
            Словарь с данными траектории или None в случае ошибки
        """
        try:
            # Генерируем случайные параметры
            params = self.config.generate_random_params(seed=trajectory_id)
            
            # Создаем решатель
            solver = DelayedMathieuEquationJIT(**params)
            
            # Решаем уравнение
            solution = solver.solve(
                t_end=self.config.t_end,
                dt=self.config.dt,
                initial_conditions=self.config.initial_conditions,
                transient_time=self.config.transient_time
            )
            
            # Проверяем качество решения
            if not self._validate_solution(solution):
                return None
            
            # Формируем результат
            trajectory_data = {
                'id': trajectory_id,
                'parameters': params,
                'solution': solution,
                'metadata': self._compute_metadata(solution)
            }
            
            return trajectory_data
            
        except Exception as e:
            logger.warning(f"Ошибка при генерации траектории {trajectory_id}: {str(e)}")
            return None
    
    def _validate_solution(self, solution: Dict) -> bool:
        """
        Проверяет качество решения
        
        Args:
            solution: Решение уравнения
            
        Returns:
            True если решение прошло валидацию
        """
        try:
            x = solution['x']
            x_dot = solution['x_dot']
            
            # Проверяем на NaN и Inf
            if np.any(np.isnan(x)) or np.any(np.isnan(x_dot)):
                return False
            
            if np.any(np.isinf(x)) or np.any(np.isinf(x_dot)):
                return False
            
            # Проверяем амплитуду
            max_amplitude = np.max(np.abs(x))
            min_amplitude = np.min(np.abs(x))
            
            if max_amplitude > self.config.filter_config['max_amplitude']:
                return False
            
            if max_amplitude < self.config.filter_config['min_amplitude']:
                return False
            
            # Проверяем стабильность (если включена)
            if self.config.filter_config['check_stability']:
                # Проверяем, что решение не взрывается экспоненциально
                last_quarter = len(x) // 4
                if last_quarter > 10:
                    recent_max = np.max(np.abs(x[-last_quarter:]))
                    early_max = np.max(np.abs(x[:last_quarter]))
                    
                    if recent_max > 10 * early_max:  # Слишком быстрый рост
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Ошибка при валидации решения: {str(e)}")
            return False
    
    def _compute_metadata(self, solution: Dict) -> Dict:
        """
        Вычисляет метаданные для решения
        
        Args:
            solution: Решение уравнения
            
        Returns:
            Словарь с метаданными
        """
        try:
            x = solution['x']
            t = solution['t']
            
            metadata = {
                'max_amplitude': float(np.max(np.abs(x))),
                'mean_amplitude': float(np.mean(np.abs(x))),
                'std_amplitude': float(np.std(x)),
                'duration': float(t[-1] - t[0]),
                'num_points': len(x),
                'sampling_rate': float(1.0 / (t[1] - t[0])) if len(t) > 1 else 0.0
            }
            
            # Дополнительные характеристики
            try:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(x, distance=20)
                if len(peaks) > 2:
                    periods = np.diff(t[peaks])
                    metadata['mean_period'] = float(np.mean(periods))
                    metadata['std_period'] = float(np.std(periods))
                    metadata['num_peaks'] = len(peaks)
                else:
                    metadata['mean_period'] = None
                    metadata['std_period'] = None
                    metadata['num_peaks'] = len(peaks)
            except ImportError:
                # scipy не установлена
                metadata['mean_period'] = None
                metadata['std_period'] = None
                metadata['num_peaks'] = None
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Ошибка при вычислении метаданных: {str(e)}")
            return {}
    
    def generate_dataset(self) -> Dict:
        """
        Генерирует полный датасет
        
        Returns:
            Словарь с данными датасета
        """
        logger.info("Начинаю генерацию датасета...")
        self.config.print_config()
        
        start_time = time.time()
        
        # Генерируем траектории параллельно
        if self.config.num_jobs == 1:
            # Последовательная генерация с прогресс-баром
            trajectories = []
            for i in tqdm(range(self.config.num_trajectories), total=self.config.num_trajectories, desc="Генерация траекторий"):
                trajectory = self.generate_single_trajectory(i)
                if trajectory is not None:
                    trajectories.append(trajectory)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                
                self.stats['total_generated'] += 1
        else:
            # Параллельная генерация
            logger.info(f"Использую {self.config.num_jobs} процессов для генерации")
            
            with Pool(processes=self.config.num_jobs) as pool:
                trajectory_ids = list(range(self.config.num_trajectories))
                # Используем tqdm для отображения прогресса
                results = []
                with tqdm(total=len(trajectory_ids), desc="Генерация траекторий") as pbar:
                    for result in pool.imap(self.generate_single_trajectory, trajectory_ids):
                        results.append(result)
                        pbar.update(1)
            
            # Фильтруем успешные результаты
            trajectories = []
            for result in results:
                if result is not None:
                    trajectories.append(result)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
                
                self.stats['total_generated'] += 1
        
        generation_time = time.time() - start_time
        self.stats['generation_time'] = generation_time
        
        # Формируем финальный датасет
        dataset = {
            'trajectories': trajectories,
            'config': self.config.__dict__,
            'stats': self.stats,
            'generation_info': {
                'timestamp': time.time(),
                'generation_time': generation_time,
                'num_successful': len(trajectories),
                'success_rate': len(trajectories) / self.config.num_trajectories if self.config.num_trajectories > 0 else 0
            }
        }
        
        logger.info(f"Генерация завершена за {generation_time:.2f} секунд")
        logger.info(f"Успешно сгенерировано: {len(trajectories)}/{self.config.num_trajectories} траекторий")
        logger.info(f"Процент успеха: {dataset['generation_info']['success_rate']:.1%}")
        
        return dataset
    
    def save_dataset(self, dataset: Dict) -> str:
        """
        Сохраняет датасет в файл
        
        Args:
            dataset: Данные датасета
            
        Returns:
            Путь к сохраненному файлу
        """
        output_path = self.config.get_output_path()
        
        logger.info(f"Сохраняю датасет в {output_path}")
        
        try:
            with open(output_path, 'wb') as f:
                if self.config.save_config['compression']:
                    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(dataset, f)
            
            # Сохраняем метаданные отдельно если нужно
            if self.config.save_config['save_metadata']:
                metadata_path = self.config.get_metadata_path()
                metadata = {
                    'config': dataset['config'],
                    'stats': dataset['stats'],
                    'generation_info': dataset['generation_info']
                }
                
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                logger.info(f"Метаданные сохранены в {metadata_path}")
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            logger.info(f"Датасет сохранен. Размер файла: {file_size:.2f} MB")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении датасета: {str(e)}")
            raise
    
    def load_dataset(self, file_path: str) -> Dict:
        """
        Загружает датасет из файла
        
        Args:
            file_path: Путь к файлу датасета
            
        Returns:
            Данные датасета
        """
        logger.info(f"Загружаю датасет из {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                dataset = pickle.load(f)
            
            logger.info(f"Датасет загружен. Траекторий: {len(dataset['trajectories'])}")
            return dataset
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета: {str(e)}")
            raise
    
    def print_dataset_info(self, dataset: Dict):
        """Выводит информацию о датасете"""
        print("\n" + "=" * 60)
        print("ИНФОРМАЦИЯ О ДАТАСЕТЕ")
        print("=" * 60)
        
        trajectories = dataset['trajectories']
        stats = dataset['stats']
        gen_info = dataset['generation_info']
        
        print(f"Количество траекторий: {len(trajectories)}")
        print(f"Время генерации: {gen_info['generation_time']:.2f} сек")
        print(f"Процент успеха: {gen_info['success_rate']:.1%}")
        
        if trajectories:
            # Статистика по параметрам
            print("\nСтатистика по параметрам:")
            param_names = list(trajectories[0]['parameters'].keys())
            
            for param in param_names:
                values = [t['parameters'][param] for t in trajectories if t['parameters'][param] is not None]
                if values:
                    print(f"  {param}: {np.mean(values):.3f} ± {np.std(values):.3f} [{np.min(values):.3f}, {np.max(values):.3f}]")
            
            # Статистика по метаданным
            print("\nСтатистика по решениям:")
            if trajectories[0]['metadata']:
                meta_keys = ['max_amplitude', 'mean_amplitude', 'duration', 'num_points']
                for key in meta_keys:
                    values = [t['metadata'][key] for t in trajectories if key in t['metadata'] and t['metadata'][key] is not None]
                    if values:
                        print(f"  {key}: {np.mean(values):.3f} ± {np.std(values):.3f}")
        
        print("=" * 60)


def worker_init():
    """Инициализация worker процесса"""
    np.random.seed()


# Функция для использования в multiprocessing
def generate_trajectory_worker(args):
    """Worker функция для генерации траектории"""
    trajectory_id, config_dict = args
    
    # Восстанавливаем конфигурацию
    config = DatasetConfig()
    config.__dict__.update(config_dict)
    
    # Создаем генератор
    generator = DatasetGenerator(config)
    
    return generator.generate_single_trajectory(trajectory_id)