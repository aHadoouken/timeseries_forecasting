#!/usr/bin/env python3
"""
Главный скрипт для генерации датасета решений уравнения Матье с запаздыванием
"""

import argparse
import sys
import os
import time
import logging
from pathlib import Path

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DatasetConfig, default_config
from dataset_generator import DatasetGenerator


def setup_logging(verbose: bool = False):
    """Настройка логирования"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dataset_generation.log')
        ]
    )


def create_custom_config(args) -> DatasetConfig:
    """Создает кастомную конфигурацию на основе аргументов командной строки"""
    config = DatasetConfig()
    
    # Обновляем параметры из аргументов командной строки
    if args.num_trajectories:
        config.num_trajectories = args.num_trajectories
    
    if args.num_jobs:
        config.num_jobs = args.num_jobs
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.dataset_name:
        config.dataset_name = args.dataset_name
    
    if args.t_end:
        config.t_end = args.t_end
    
    if args.dt:
        config.dt = args.dt
    
    if args.transient_time:
        config.transient_time = args.transient_time
    
    return config


def generate_dataset_command(args):
    """Команда генерации датасета"""
    print("🚀 Запуск генерации датасета...")
    
    # Создаем конфигурацию
    config = create_custom_config(args)
    
    # Создаем генератор
    generator = DatasetGenerator(config)
    
    try:
        # Генерируем датасет
        dataset = generator.generate_dataset()
        
        # Сохраняем датасет
        output_path = generator.save_dataset(dataset)
        
        # Выводим информацию о датасете
        generator.print_dataset_info(dataset)
        
        print(f"\n✅ Датасет успешно сгенерирован и сохранен в: {output_path}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Ошибка при генерации датасета: {str(e)}")
        return 1


def info_dataset_command(args):
    """Команда для получения информации о датасете"""
    if not os.path.exists(args.dataset_path):
        print(f"❌ Файл датасета не найден: {args.dataset_path}")
        return 1
    
    try:
        # Создаем генератор для загрузки
        config = DatasetConfig()
        generator = DatasetGenerator(config)
        
        # Загружаем датасет
        dataset = generator.load_dataset(args.dataset_path)
        
        # Выводим информацию
        generator.print_dataset_info(dataset)
        
        return 0
        
    except Exception as e:
        logging.error(f"Ошибка при загрузке датасета: {str(e)}")
        return 1


def validate_config_command(args):
    """Команда для валидации конфигурации"""
    try:
        config = create_custom_config(args)
        config.validate_config()
        config.print_config()
        print("\n✅ Конфигурация корректна!")
        return 0
        
    except Exception as e:
        print(f"❌ Ошибка в конфигурации: {str(e)}")
        return 1


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Генератор датасета решений уравнения Матье с запаздыванием",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Генерация датасета с параметрами по умолчанию
  python main.py generate

  # Генерация 500 траекторий с использованием 8 процессов
  python main.py generate --num-trajectories 500 --num-jobs 8

  # Генерация с кастомными параметрами
  python main.py generate --num-trajectories 1000 --t-end 3000 --output-dir my_datasets

  # Получение информации о существующем датасете
  python main.py info datasets/mathieu_delayed_dataset.pkl

  # Валидация конфигурации
  python main.py validate --num-trajectories 100 --num-jobs 4
        """
    )
    
    # Общие аргументы
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    
    # Создаем подкоманды
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Команда генерации
    generate_parser = subparsers.add_parser('generate', help='Генерировать датасет')
    generate_parser.add_argument('--num-trajectories', type=int,
                                help=f'Количество траекторий (по умолчанию: {default_config.num_trajectories})')
    generate_parser.add_argument('--num-jobs', type=int,
                                help=f'Количество процессов (по умолчанию: {default_config.num_jobs})')
    generate_parser.add_argument('--output-dir', type=str,
                                help=f'Директория для сохранения (по умолчанию: {default_config.output_dir})')
    generate_parser.add_argument('--dataset-name', type=str,
                                help=f'Имя датасета (по умолчанию: {default_config.dataset_name})')
    generate_parser.add_argument('--t-end', type=float,
                                help=f'Время интегрирования (по умолчанию: {default_config.t_end})')
    generate_parser.add_argument('--dt', type=float,
                                help=f'Шаг интегрирования (по умолчанию: {default_config.dt})')
    generate_parser.add_argument('--transient-time', type=float,
                                help=f'Время переходного процесса (по умолчанию: {default_config.transient_time})')
    
    # Команда информации
    info_parser = subparsers.add_parser('info', help='Получить информацию о датасете')
    info_parser.add_argument('dataset_path', type=str,
                            help='Путь к файлу датасета')
    
    # Команда валидации
    validate_parser = subparsers.add_parser('validate', help='Валидировать конфигурацию')
    validate_parser.add_argument('--num-trajectories', type=int,
                                help='Количество траекторий для проверки')
    validate_parser.add_argument('--num-jobs', type=int,
                                help='Количество процессов для проверки')
    validate_parser.add_argument('--output-dir', type=str,
                                help='Директория для сохранения для проверки')
    validate_parser.add_argument('--dataset-name', type=str,
                                help='Имя датасета для проверки')
    validate_parser.add_argument('--t-end', type=float,
                                help='Время интегрирования для проверки')
    validate_parser.add_argument('--dt', type=float,
                                help='Шаг интегрирования для проверки')
    validate_parser.add_argument('--transient-time', type=float,
                                help='Время переходного процесса для проверки')
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Настраиваем логирование
    setup_logging(args.verbose)
    
    # Выполняем команду
    if args.command == 'generate':
        return generate_dataset_command(args)
    elif args.command == 'info':
        return info_dataset_command(args)
    elif args.command == 'validate':
        return validate_config_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Генерация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Неожиданная ошибка: {str(e)}")
        sys.exit(1)