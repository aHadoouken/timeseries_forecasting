#!/usr/bin/env python3
"""
Пример использования генератора датасета
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DatasetConfig
from dataset_generator import DatasetGenerator
from equation_solver import DelayedMathieuEquationJIT


def example_single_trajectory():
    """Пример генерации одной траектории"""
    print("=" * 60)
    print("ПРИМЕР: Генерация одной траектории")
    print("=" * 60)
    
    # Параметры из test6
    params = {
        'k': 0.182, 'd': 2.5, 'e': 0, 'T': 2*np.pi,
        'b': 0.37, 'tau': 2*np.pi, 'nonlin_k': 0.04,
        "b_ts": 2000, "b_te": 2300, "b_e": 0.42,
        'F_ts': 0, 'F_te': 10, 'F_As': 4, 'F_Ae': 4, 'F_T': 2*np.pi/3
    }
    
    # Создаем решатель
    solver = DelayedMathieuEquationJIT(**params)
    
    # Решаем уравнение
    solution = solver.solve(t_end=1000, dt=0.01, transient_time=100)
    
    print(f"Решение получено:")
    print(f"  Время: {solution['t'][0]:.1f} - {solution['t'][-1]:.1f}")
    print(f"  Точек: {len(solution['t'])}")
    print(f"  Амплитуда x: {np.min(solution['x']):.3f} - {np.max(solution['x']):.3f}")
    print(f"  Амплитуда x': {np.min(solution['x_dot']):.3f} - {np.max(solution['x_dot']):.3f}")
    
    return solution


def example_small_dataset():
    """Пример генерации небольшого датасета"""
    print("\n" + "=" * 60)
    print("ПРИМЕР: Генерация небольшого датасета")
    print("=" * 60)
    
    # Создаем конфигурацию для небольшого датасета
    config = DatasetConfig()
    config.num_trajectories = 10  # Только 10 траекторий для примера
    config.num_jobs = 2  # 2 процесса
    config.t_end = 1000  # Короткое время интегрирования
    config.dataset_name = "example_small_dataset"
    config.output_dir = "example_datasets"
    
    # Создаем генератор
    generator = DatasetGenerator(config)
    
    # Генерируем датасет
    dataset = generator.generate_dataset()
    
    # Сохраняем датасет
    output_path = generator.save_dataset(dataset)
    
    # Выводим информацию
    generator.print_dataset_info(dataset)
    
    return dataset, output_path


def example_load_and_analyze():
    """Пример загрузки и анализа датасета"""
    print("\n" + "=" * 60)
    print("ПРИМЕР: Загрузка и анализ датасета")
    print("=" * 60)
    
    # Сначала создаем небольшой датасет
    dataset, dataset_path = example_small_dataset()
    
    # Загружаем датасет
    config = DatasetConfig()
    generator = DatasetGenerator(config)
    loaded_dataset = generator.load_dataset(dataset_path)
    
    # Анализируем параметры
    trajectories = loaded_dataset['trajectories']
    
    print(f"\nАнализ параметров ({len(trajectories)} траекторий):")
    
    # Собираем статистику по параметрам
    param_stats = {}
    for param_name in trajectories[0]['parameters'].keys():
        values = []
        for traj in trajectories:
            val = traj['parameters'][param_name]
            if val is not None and not isinstance(val, str):
                values.append(val)
        
        if values:
            param_stats[param_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    for param, stats in param_stats.items():
        print(f"  {param}: {stats['mean']:.3f} ± {stats['std']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Анализ решений
    print(f"\nАнализ решений:")
    amplitudes = [traj['metadata']['max_amplitude'] for traj in trajectories]
    durations = [traj['metadata']['duration'] for traj in trajectories]
    
    print(f"  Максимальные амплитуды: {np.mean(amplitudes):.3f} ± {np.std(amplitudes):.3f}")
    print(f"  Длительности: {np.mean(durations):.1f} ± {np.std(durations):.1f}")
    
    return loaded_dataset


def example_plot_trajectories(dataset, num_plots=3):
    """Пример построения графиков траекторий"""
    print(f"\n" + "=" * 60)
    print(f"ПРИМЕР: Построение графиков {num_plots} траекторий")
    print("=" * 60)
    
    trajectories = dataset['trajectories']
    
    # Выбираем случайные траектории для отображения
    indices = np.random.choice(len(trajectories), min(num_plots, len(trajectories)), replace=False)
    
    fig, axes = plt.subplots(num_plots, 2, figsize=(15, 4*num_plots))
    if num_plots == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        traj = trajectories[idx]
        solution = traj['solution']
        params = traj['parameters']
        
        # График x(t)
        axes[i, 0].plot(solution['t'], solution['x'], 'b-', linewidth=1)
        axes[i, 0].set_xlabel('Время t')
        axes[i, 0].set_ylabel('x(t)')
        axes[i, 0].set_title(f'Траектория {idx}: k={params["k"]:.3f}, d={params["d"]:.3f}')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Фазовый портрет
        axes[i, 1].plot(solution['x'], solution['x_dot'], 'r-', linewidth=1, alpha=0.7)
        axes[i, 1].set_xlabel('x(t)')
        axes[i, 1].set_ylabel("x'(t)")
        axes[i, 1].set_title(f'Фазовый портрет {idx}')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Графики сохранены в example_trajectories.png")


def main():
    """Главная функция с примерами"""
    print("🚀 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ГЕНЕРАТОРА ДАТАСЕТА")
    print("=" * 80)
    
    try:
        # Пример 1: Одна траектория
        solution = example_single_trajectory()
        
        # Пример 2: Небольшой датасет
        dataset, _ = example_small_dataset()
        
        # Пример 3: Загрузка и анализ
        loaded_dataset = example_load_and_analyze()
        
        # Пример 4: Построение графиков (если matplotlib доступен)
        try:
            example_plot_trajectories(loaded_dataset, num_plots=3)
        except ImportError:
            print("\n⚠️  matplotlib не установлен, пропускаю построение графиков")
        except Exception as e:
            print(f"\n⚠️  Ошибка при построении графиков: {e}")
        
        print("\n" + "=" * 80)
        print("✅ Все примеры выполнены успешно!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Ошибка в примерах: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()