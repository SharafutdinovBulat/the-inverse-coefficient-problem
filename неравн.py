from numpy import zeros, linspace, tanh, complex64, sin, ones, transpose, array, power, pi, ravel, concatenate
from matplotlib.pyplot import style, figure, axes, plot, text, legend, subplots, show, xlabel, ylabel, title, show,grid, ylim, xlim
from celluloid import Camera
import json
import scipy.integrate as spi
import matplotlib.animation as animation

# Модельная функция
def q_theor(x):
    return sin(3 * pi * x)
# Плотность распределения x(d)
def x_cubic(d):
    return A * d ** 3 + x_0
# Решение прямой задачи, возвращает массив U
def solution_straight_problem(q):
    def u_init(x):
        u_init = 0.5*tanh((x - x_0)/eps)
        return u_init

    # Определение функции, задающей левое граничное условие
    def u_left(t):
        u_left = -0.5
        return u_left
    # Определение функции, задающей правое граничное условие

    def u_right(t):
        u_right = 0.5
        return u_right
    # Функция f подготавливает массив, содержащий элементы вектор-функции,
    # определяющей правую часть решаемой системы ОДУ

    def f(y, t, N, u_left, u_right, eps, q):
        f = zeros(N-1)
        f[0] = eps*(y[1] - 2*y[0] + u_left(t))/h[0]**2 + y[0] * \
                    (y[1] - u_left(t))/(2*h[0]) - y[0] * q[0]
        for n in range(1, N-2):
            f[n] = eps*(y[n+1] - 2*y[n] + y[n-1])/h[n]**2 + y[n] * \
                        (y[n+1] - y[n-1])/(2*h[n]) - y[n] * q[n]
        f[N-2] = eps*(u_right(t) - 2*y[N-2] + y[N-3])/h[n]**2 + \
                      y[N-2]*(u_right(t) - y[N-3])/(2*h[n]) - y[N-2] * q[N-2]
        return f

    def DiagonalsPreparation(y, t, N, u_left, u_right, eps, tau, alpha, q):
        a = zeros(N-1, dtype=complex64)
        b = zeros(N-1, dtype=complex64)
        c = zeros(N-1, dtype=complex64)
        a[0] = 1. - alpha*tau*(-2*eps/h[0]**2 + (y[1] - u_left(t))/(2*h[0]) - q[0])
        c[0] = - alpha*tau*(eps/h[0]**2 + y[0]/(2*h[0]))
        for n in range(1, N-2):
            b[n] = - alpha*tau*(eps/h[n]**2 - y[n]/(2*h[n]))
            a[n] = 1. - alpha*tau * \
                (-2*eps/h[n]**2 + (y[n+1] - y[n-1])/(2*h[n]) - q[n])
            c[n] = - alpha*tau*(eps/h[n]**2 + y[n]/(2*h[n]))
        b[N-2] = - alpha*tau*(eps/h[N-2]**2 - y[N-2]/(2*h[N-2]))
        a[N-2] = 1. - alpha*tau * \
            (-2*eps/h[N-2]**2 + (u_right(t) - y[N-3])/(2*h[N-2]) - q[N-2])
        return a, b, c

    def TridiagonalMatrixAlgorithm(a, b, c, B):
        n = len(B)
        v = zeros(n, dtype=complex64)
        X = zeros(n, dtype=complex64)
        w = a[0]
        X[0] = B[0]/w
        for i in range(1, n):
            v[i - 1] = c[i - 1]/w
            w = a[i] - b[i]*v[i - 1]
            X[i] = (B[i] - b[i]*X[i - 1])/w
        for j in range(n-2, -1, -1):
            X[j] = X[j] - v[j]*X[j + 1]
        return X

    def PDESolving(a, b, N, t_0, T, M, u_init, u_left, u_right, eps, alpha):
        x = linspace(a, b, N+1)
        tau = (T - t_0)/M; t = linspace(t_0, T, M+1)
        u = zeros((M + 1, N + 1))
        y = zeros(N - 1)
        u[0] = u_init(x)
        y = u_init(x[1:N])

        for m in range(M):
            diagonal, codiagonal_down, codiagonal_up = DiagonalsPreparation(
                y, t[m], N, u_left, u_right, eps, tau, alpha, q)
            w_1 = TridiagonalMatrixAlgorithm(diagonal, codiagonal_down, codiagonal_up, f(
                y, t[m] + tau/2, N, u_left, u_right, eps, q))

            y = y + tau*w_1.real
            u[m + 1, 0] = u_left(t[m+1])
            u[m + 1, 1:N] = y
            u[m + 1, N] = u_right(t[m+1])
        return u
    return PDESolving(a, b, N, t_0, T, M, u_init, u_left, u_right, eps, alpha)

# Решение сопряженной задачи, возвращает массив psi
def solution_conj_problem(q, u_xT, f_obs):
    def psi_left(t):
        u_left = 0
        return u_left
    # Определение функции, задающей правое граничное условие

    def psi_right(t):
        u_right = 0
        return u_right
    # Функция f подготавливает массив, содержащий элементы вектор-функции,
    # определяющей правую часть решаемой системы ОДУ

    def g(v, t, N, psi_left, psi_right, eps, q, y):
        g = zeros(N-1)
        g[0] = -eps*(v[1] - 2*v[0] + psi_left(t))/h[0]**2 + y[0] * \
                     (v[1] - psi_left(t))/(2*h[0]) + v[0] * q[0]
        for n in range(1, N-2):
            g[n] = -eps*(v[n+1] - 2*v[n] + v[n-1])/h[n]**2 + y[n] * \
                         (v[n+1] - v[n-1])/(2*h[n]) + v[n] * q[n]
        g[N-2] = -eps*(psi_right(t) - 2*v[N-2] + v[N-3])/h[N-2]**2 + \
                       y[N-2]*(psi_right(t) - v[N-3])/(2*h[N-2]) + v[N-2] * q[N-2]
        return g

    def DiagonalsPreparation_g(y, t, N, psi_left, psi_right, eps, tau, alpha, q):
        a = zeros(N-1, dtype=complex64)
        b = zeros(N-1, dtype=complex64)
        c = zeros(N-1, dtype=complex64)
        a[0] = 1. - alpha*tau*(2*eps/h[0]**2 + q[0])
        c[0] = - alpha*tau*(-eps/h[0]**2 + y[0]/(2*h[0]))
        for n in range(1, N-2):
            b[n] = - alpha*tau*(-eps/h[n]**2 - y[n]/(2*h[n]))
            a[n] = 1. - alpha*tau*(2*eps/h[n]**2 + q[n])
            c[n] = - alpha*tau*(-eps/h[n]**2 + y[n]/(2*h[n]))
        b[N-2] = - alpha*tau*(-eps/h[N-2]**2 - y[N-2]/(2*h))
        a[N-2] = 1. - alpha*tau*(2*eps/h[N-2]**2 + q[N-2])
        return a, b, c

    def TridiagonalMatrixAlgorithm(a, b, c, B):
        n = len(B)
        v = zeros(n, dtype=complex64)
        X = zeros(n, dtype=complex64)
        w = a[0]
        X[0] = B[0]/w
        for i in range(1, n):
            v[i - 1] = c[i - 1]/w
            w = a[i] - b[i]*v[i - 1]
            X[i] = (B[i] - b[i]*X[i - 1])/w
        for j in range(n-2, -1, -1):
            X[j] = X[j] - v[j]*X[j + 1]
        return X

    def PDESolving_g(a, b, N, t_0, T, M, psi_left, psi_right, eps, alpha):
        x = linspace(a, b, N+1)
        tau = (T - t_0)/M; t = linspace(t_0, T, M+1)
        psi = zeros((M + 1, N + 1))
        v = zeros(N - 1)
        psi[-1] = -2 * (u_xT - f_obs)
        y = (psi[-1])[1:N]

        for m in range(0, M, - 1):
            diagonal, codiagonal_down, codiagonal_up = DiagonalsPreparation_g(
                y, t, N, psi_left, psi_right, eps, tau, alpha, q)
            w_1 = TridiagonalMatrixAlgorithm(diagonal, codiagonal_down, codiagonal_up, g(
                v, t[m] - tau / 2, N, psi_left, psi_right, eps, q, y))

            v = v - tau*w_1.real
            psi[m - 1, 0] = psi_left(t[m-1])
            psi[m - 1, 1:N] = y
            psi[m - 1, N] = psi_right(t[m-1])
        return psi
    return PDESolving_g(a, b, N, t_0, T, M, psi_left, psi_right, eps, alpha)

def Q_animation(animation_bool, name, step):
    q1 = q_theor(x)
    fig, ax = subplots()
    line, = ax.plot([], [])
    s1 = [i for i in range(0, quantity_of_iterations+quantity_of_iterations_0, step)]
    
    
    def animate(i):
        ax.clear()
        ax.plot(x, Q[i, :], color='r')
        ax.plot(x, q1, color = 'g')
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.5, 3)
        xlabel('x')
        ylabel('q')
        title('Анимация графика решения q(x) в зависимости от итерации\n N = {}, M = {}, beta = {}, итерация {}'.format(N, M , beta, i))
        grid()
        legend(['Восстановленная', 'Модельная'], loc = 'upper left')
        return line,
    
    
    ani = animation.FuncAnimation(fig, animate, frames=s1, interval=10, blit=True)
    if len(name) == 0:    
        name = '_'.join([str(N), str(M), str(quantity_of_iterations + quantity_of_iterations_0), str(beta)]) + '.gif'
    else:
        name += '.gif'
    ani.save(name, writer='pillow')
    
# Стационарные величины
a = 0.; b = 1.
t_0 = 0.; T = 6.0
x_0 = 0.6
eps = 10**(-2.0)
alpha_reg =  0 * 10**(-3.0)
quantity_of_iterations_0 = 0

#Переменные
alpha = (1 + 1j)/2
N1 = 60
N = 3 * N1; M = 20
A = 0.01
quantity_of_iterations = 500
beta = 1.0
grid_type = 2           # 0 - равномерная, 1 - прямые, 2 - куб
version = 0             # 0 - новый файл, 1 - чтение другого файла  
animation_bool = 0      # 0 - не сохраняем анимацию, 1 - сохраняем анимацию
animation_file_name = 'bababui' # Опциональный параметр, по умолчанию файл сохраняется в формате 'N_M_quantity of iter_beta_txt'
file_name = '90_20_4000_1.0_.txt'  # имя файла для чтения
step = 10 # q анимируется каждый step шагов


# Режим работы программ
match grid_type:
    
    
    case 0:
        #Равномерная
        x = linspace(a, b, N+1)
        h = (b-a) / N * ones(N)
    case 1:
        # Много линий
        x = concatenate([linspace(a, x_0 - 3 * eps, N1 ), linspace(x_0 - 3 * eps, x_0 + 3 * eps, N1 ), linspace(x_0 + 3 * eps, b, N1 + 1)])
    case 2:
        #По кубу
        d = linspace(-(x_0 / A) ** (1/3),((1-x_0) / A) ** (1/3), N+1)
        x = x_cubic(d)
        x1 = x[1:]
        h = (x1 - x[:N])
match version:
    
    case 0:
        J_array = zeros(quantity_of_iterations)
        Q = zeros((quantity_of_iterations, N+1))
        summa = zeros(quantity_of_iterations)
        q = zeros(N+1)
    case 1:
        with open(file_name, 'r') as fr:
            # читаем из файла
            lst = json.load(fr)
            

        quantity_of_iterations_0, u_T, J, q, summa1 = lst
        u_T, J, q, summa1 = array(u_T),array(J),array(q),array(summa1)
        
        J_array = zeros(quantity_of_iterations + quantity_of_iterations_0)
        Q = zeros((quantity_of_iterations+quantity_of_iterations_0, N+1))
        summa = zeros(quantity_of_iterations+quantity_of_iterations_0)
        
        J_array[0:quantity_of_iterations_0] = J
        summa[0:quantity_of_iterations_0] = summa1
        
        
        
# Промежуточные операции
tau = (T - t_0)/M; t = linspace(t_0, T, M+1) 
file_name_Q = "Q" + file_name
q1 = q_theor(x)
u = solution_straight_problem(q1)
f_obs = u[-1]



# Основной цикл, в котором ищется q[s]
for i in range(quantity_of_iterations_0, quantity_of_iterations + quantity_of_iterations_0):
    u = solution_straight_problem(q)
    psi = solution_conj_problem(q, u[-1], f_obs)
    mult_of_array = transpose(u * psi)
    J = array([spi.simpson(j, t) for j in mult_of_array]) + 2 * alpha_reg * q
    q = q - beta * J
    Q[i] = q
    summa[i] = spi.simpson(abs(q - q1), x)
    u_T = (solution_straight_problem(q))[-1]
    variable = power((u_T - f_obs), 2)
    variable2 = sum(q**2)
    #J_array[i] = sum(variable) * h + variable2 * h * alpha_reg
    J_array[i] = spi.simpson(variable, x) + alpha_reg * spi.simpson(q ** 2, x)
    print(i, J_array[i], beta)


# Подготовка графиков
s = [i for i in range(quantity_of_iterations+quantity_of_iterations_0)]

# =============================================================================
# fig, axs = subplots(nrows= 1 , ncols= 2 )
# axs[0].set_xlabel('x'); axs[0].set_ylabel('q(x)')
# axs[0].plot(x,q)
# axs[0].title('ds')
q1 = q_theor(x)
# axs[0].plot(x, q1)
# # axs[0].legend(['Восстановленная', 'Теоретическая'], loc = 1)
# axs[1].plot(s,J_array)
# =============================================================================
plot(x, q, color = 'r')
plot(x, q1, color = 'g')
xlim(a, b)
grid()
title('График зависимости q(x) для N = {}, M = {}, beta = {}, s = {}'.format(N, M, beta, quantity_of_iterations + quantity_of_iterations_0))
legend(['Восстановленная', 'Модельная'])
show()



Q_animation(animation_bool, animation_file_name, step)



file_name = '_'.join([str(N), str(M), str(quantity_of_iterations + quantity_of_iterations_0), str(beta)]) + '_.txt'
print(file_name)

file = [quantity_of_iterations + quantity_of_iterations_0, list(u_T), list(J_array), list(q), list(summa)]

with open(file_name, 'w') as fw:
    # записываем
    json.dump(file, fw)   





