"""
Упрощенная версия модуля для решения модифицированного уравнения Матье с запаздыванием
Использует более простые функции без Piecewise для надежности
"""

import numpy as np
from jitcdde import jitcdde, y, t
from symengine import cos


class DelayedMathieuEquationSimple:
    """
    Упрощенное решение модифицированного уравнения Матье с запаздыванием:
    x''(t) + k*x'(t) + (d + e*cos(2π*t/T))*x(t) = b*x(t-τ)
    """
    
    def __init__(self, k, d, e, T, b, tau, nonlin_k=None, 
                 F_ts=None, F_te=None, F_As=None, F_Ae=None, F_T=None,
                 b_ts=None, b_te=None, b_e=None):
        self.k = k
        self.d = d
        self.e = e
        self.T = T
        self.b = b
        self.tau = tau
        self.nonlin_k = nonlin_k if nonlin_k is not None else 0.0

        # Упрощаем внешнюю силу - используем только постоянную амплитуду
        self.F_A = F_As if F_As is not None else 0.0
        self.F_T = F_T if F_T is not None else 2*np.pi
        
        # Упрощаем изменение коэффициента b - используем только постоянное значение
        self.b_final = b_e if b_e is not None else b

    def get_equations(self):
        """Получает систему уравнений в упрощенном виде"""
        # Упрощенная внешняя сила (постоянная амплитуда)
        force = self.F_A * cos(2*np.pi*t/self.F_T)
        
        # Упрощенный коэффициент запаздывания (постоянный)
        delayed_term = self.b * y(0, t-self.tau)
        
        # Основной осциллятор
        oscillator = (self.d + self.e*cos(2*np.pi*t/self.T))*y(0)
        
        # Демпфирование (с возможной нелинейностью)
        if self.nonlin_k > 0:
            damping = self.k*(1 + self.nonlin_k * y(0)**2)*y(1)
        else:
            damping = self.k*y(1)
        
        # Система уравнений
        f = [
            y(1),  # x' = y
            force + delayed_term - damping - oscillator  # y' = ...
        ]
        
        return f

    def solve(self, t_end=100, dt=0.01, initial_conditions=None, transient_time=100):
        """
        Решает уравнение используя jitcdde с Python backend
        
        Args:
            t_end: время интегрирования
            dt: шаг интегрирования
            initial_conditions: начальные условия [x(0), x'(0)]
            transient_time: время переходного процесса
        """
        if initial_conditions is None:
            initial_conditions = [1.0, 0.0]
        
        try:
            # Создаем объект DDE
            dde = jitcdde(self.get_equations(), delays=[self.tau])
            
            # Устанавливаем начальные условия
            dde.constant_past(initial_conditions)
            
            # Параметры интегрирования
            dde.set_integration_parameters(rtol=1e-6, atol=1e-8)
            
            # Принудительно используем Python backend
            dde.generate_lambdas()
            dde.adjust_diff()

            # Пропускаем переходный процесс
            if transient_time > 0:
                dde.integrate_blindly(transient_time, dt)
            
            times = []
            solution = []
            
            # Интегрируем с правильным порядком
            current_time = transient_time if transient_time > 0 else 0
            end_time = current_time + t_end
            
            while current_time < end_time:
                # Интегрируем и сохраняем результат
                state = dde.integrate(current_time)
                times.append(current_time - (transient_time if transient_time > 0 else 0))
                solution.append(state)
                current_time += dt
            
            solution = np.array(solution)
            times = np.array(times)
            
            return {
                't': times,
                'x': solution[:, 0],
                'x_dot': solution[:, 1]
            }
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при решении уравнения: {str(e)}")

    def get_parameters(self):
        """Возвращает все параметры уравнения в виде словаря"""
        return {
            'k': self.k,
            'd': self.d,
            'e': self.e,
            'T': self.T,
            'b': self.b,
            'tau': self.tau,
            'nonlin_k': self.nonlin_k,
            'F_A': self.F_A,
            'F_T': self.F_T,
            'b_final': self.b_final
        }


# Алиас для совместимости
DelayedMathieuEquationJIT = DelayedMathieuEquationSimple