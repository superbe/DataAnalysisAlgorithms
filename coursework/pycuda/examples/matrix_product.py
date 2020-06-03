import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit

# Функция ядро расчета, реализующее произведения двух матриц.
# threadIdx.x и threadIdy.y это координаты, которые позволяют указывать конкретные потоки в имеющейся сетке
# двухмерных блоков. Такие потоки внутри установленной сетки блоков выполняют один и тот же код ядра,
# но с разными частями данных. Если сравнить параллельную версию с последовательной, можно заметить,
# что индексы циклов i и j были заменены на соответствующие индексы threadIdx.x и threadIdy.y. Это означает,
# что в параллельной версии у будет одна итерация цикла. На практике ядро MatrixMulKernel будет выполнено сеткой
# с размерностью потоков 5 х 5.
kernel_code_template = """ 
__global__ void MatrixMulKernel(float *a, float *b, float *c) 
{ 
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    float Pvalue = 0; 
    for (int k = 0; k < %(MATRIX_SIZE)s; ++k) { 
        float Aelement = a[ty * %(MATRIX_SIZE)s + k]; 
        float Belement = b[k * %(MATRIX_SIZE)s + tx]; 
        Pvalue += Aelement * Belement; 
    } 
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue; 
}"""

# Задали размер матриц.
MATRIX_SIZE = 5

# Задали исходные матрицы.
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# Сделали расчет на CPU.
c_cpu = np.dot(a_cpu, b_cpu)

# Выделили память на устройстве GPU для наших матриц.
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

# Выделили память на устройстве GPU для итоговой матрицы.
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

# Переопределили ядро расчета с установленным размером matrix_size
kernel_code = kernel_code_template % {'MATRIX_SIZE': MATRIX_SIZE}

# Сформировали исполнительный модуль.
mod = compiler.SourceModule(kernel_code)

# Указали исполняемую функцию.
matrixmul = mod.get_function("MatrixMulKernel")

# Запустили функцию ядра для расчета произведения двух матриц a_gpu и b_gpu, получая в результате матрицу c_gpu.
# Значение размера блока потоков определяется как MATRIX_SIZE, MATRIX_SIZE, 1.
matrixmul(a_gpu, b_gpu, c_gpu, block=(MATRIX_SIZE, MATRIX_SIZE, 1))

# Выводим результат расчета.
print("-" * 80)
print("Matrix A (GPU):")
print(a_gpu.get())
print("-" * 80)
print("Matrix B (GPU):")
print(b_gpu.get())
print("-" * 80)
print("Matrix C (GPU):")
print(c_gpu.get())
print("-" * 80)
print("Matrix C (CPU):")
print(c_cpu)

# Проверили точночть расчета
np.allclose(c_cpu, c_gpu.get())
