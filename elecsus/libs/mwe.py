import sympy
import symengine
import pickle
with open('matrix_Ab.dat', 'rb') as f:
    A, b = pickle.load(f)
A = symengine.Matrix(A)
b = symengine.Matrix(b)
print('solve...')
x = A.LUsolve(b)
print('solved!')
print(f'Symbolic operations: {symengine.count_ops(x)}')
# v = symengine.var('w_01 Efield_01')
# f = symengine.Lambdify([v], x, real=False, cse=True)
