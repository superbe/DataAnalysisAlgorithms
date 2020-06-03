# Импортировали драйвер устройства.
import pycuda.driver as drv
# Определили какой именно GPU в нашей системе доступен для работы
import pycuda.autoinit
# Директива компилятора nVidia (nvcc).
from pycuda.compiler import SourceModule
import numpy

# Сформировали матрицу 5 x 5, значения которой выбираются случайным образом.
a = numpy.random.randn(5, 5)
# GPU поддерживает только одинарную точность.
a = a.astype(numpy.float32)

# Выделили память на устройстве для нашей матрицы.
a_gpu = drv.mem_alloc(a.nbytes)

# Перенесли матрицу на устройство.
drv.memcpy_htod(a_gpu, a)

# Сформировали исполнительный модуль.
mod = SourceModule(""" 
  __global__ void doubles_matrix(float *a){ 
    int idx = threadIdx.x + threadIdx.y*4; 
    a[idx] *= 2;} 
  """)

# Указали исполняемую функцию.
func = mod.get_function("doubles_matrix")

# Запустили функцию ядра.
func(a_gpu, block=(5, 5, 1))

# Выделили область памяти для скопированной матрицы.
a_doubled = numpy.empty_like(a)

# Перенесли новую матрицу с устройства в оперативную память
drv.memcpy_dtoh(a_doubled, a_gpu)

# Отобразили результат.
print("ORIGINAL MATRIX")
print(a)
print("DOUBLED MATRIX AFTER PyCUDA EXECUTION")
print(a_doubled)
