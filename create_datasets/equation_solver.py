"""
Модуль для решения модифицированного уравнения Матье с запаздыванием
Скопировано и адаптировано из test.ipynb
"""

import numpy as np
from jitcdde import jitcdde, y, t
from symengine import cos, Piecewise


class DelayedMathieuEquationJIT:
    """
    Решение модифицированного уравнения Матье с запаздыванием:
    x''(t) + k*x'(t) + (d + e*cos(2π*t/T))*x(t) = b*x(t-τ)
    """
    
    def __init__(self, k, d, e, T, b, tau,
                 nonlin_k=None, F_ts=None, F_te=None, F_As=None, F_Ae=None, F_T=None,
                 b_ts=None, b_te=None, b_e=None):
        self.k = k
        self.d = d
        self.e = e
        self.T = T
        self.b = b
        self.tau = tau
        self.nonlin_k = nonlin_k

        self.F_ts = F_ts
        self.F_te = F_te
        self.F_As = F_As
        self.F_Ae = F_Ae
        self.F_T = F_T

        self.b_ts = b_ts
        self.b_te = b_te
        self.b_e = b_e

    def force(self, t):
        if self.F_ts is None or self.F_te is None or self.F_As is None or self.F_Ae is None or self.F_T is None:
            return 0
        
        c = cos(2*np.pi*t/self.F_T)
        r = Piecewise(
            (self.F_As * c, t < self.F_ts),
            (self.F_Ae * c, t > self.F_te),
            ((self.F_As + (t - self.F_ts) / (self.F_te - self.F_ts) * (self.F_Ae - self.F_As)) * c, True)
        )
        return r
    
    def calc_b(self):
        if self.b_ts is None or self.b_te is None or self.b_e is None:
            return self.b
        return Piecewise(
            (self.b, t < self.b_ts),
            (self.b_e, t > self.b_te),
            ((self.b + (t - self.b_ts) / (self.b_te - self.b_ts) * (self.b_e - self.b)), True)
        )
    
    def get_equations(self):
        delayed_term = self.calc_b()*y(0, t-self.tau)
        oscillator = (self.d + self.e*cos(2*np.pi*t/self.T))*y(0)
        if self.nonlin_k:
            damping = self.k*(1 + self.nonlin_k * y(0)**2)*y(1)
        else:
            damping = self.k*y(1)
        f = [
            y(1),
            self.force(t) + delayed_term - damping - oscillator
        ]
        return f

    def solve(self, t_end=100, dt=0.01, initial_conditions=None, transient_time=100):
        """
        Решает уравнение используя jitcdde
        
        initial_conditions: [x(0), x'(0)]
        """
        if initial_conditions is None:
            initial_conditions = [1, 0.0]
        
        # Создаем объект DDE
        dde = jitcdde(self.get_equations(), delays=[self.tau])
        
        # Устанавливаем начальные условия
        dde.constant_past(initial_conditions)
        
        # Параметры интегрирования
        dde.set_integration_parameters(rtol=1e-8, atol=1e-10)
        
        # Используем Python backend вместо компиляции C кода для надежности
        try:
            dde.adjust_diff()
        except:
            # Если компиляция не удалась, используем lambdified функции
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
            'F_ts': self.F_ts,
            'F_te': self.F_te,
            'F_As': self.F_As,
            'F_Ae': self.F_Ae,
            'F_T': self.F_T,
            'b_ts': self.b_ts,
            'b_te': self.b_te,
            'b_e': self.b_e
        }