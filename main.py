from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Literal
from fractions import Fraction
from fastapi.middleware.cors import CORSMiddleware
import copy

app = FastAPI(title="Simplex Solver API")

# Añadir el middleware CORS a la aplicación FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # Permitir todos los encabezados
)

def parse_fraction_dict(d):
    return {k: Fraction(v) if k != "sign" else v for k, v in d.items()}

def fraction_to_str(d):
    res = {}
    for k,v in d.items():
        if isinstance(v, Fraction):
            res[k] = str(v)
        else:
            res[k] = v
    return res

def table_row_to_dict(row, variables):
    res = {}
    for key in ['Ci', 'Vb', 'Bi', 'Øi'] + variables:
        val = row.get(key, '')
        if val is None:
            val = ''
        if isinstance(val, Fraction):
            val = str(val)
        res[key] = val
    return res

def standardize_objective_function(obj_func, restrictions, optimization='max', M=1000):
    updated_obj = obj_func.copy()
    slack_count = 1
    artificial_count = 1
    for restr in restrictions:
        sign = restr.get("sign")
        if sign == "<=":
            slack_var = f"S{slack_count}"
            updated_obj[slack_var] = Fraction(0)
            slack_count += 1
        elif sign == ">=":
            slack_var = f"S{slack_count}"
            artificial_var = f"A{artificial_count}"
            updated_obj[slack_var] = Fraction(0)
            updated_obj[artificial_var] = Fraction(-M) if optimization == 'max' else Fraction(M)
            slack_count += 1
            artificial_count += 1
        elif sign == "=":
            artificial_var = f"A{artificial_count}"
            updated_obj[artificial_var] = Fraction(-M) if optimization == 'max' else Fraction(M)
            artificial_count += 1
        else:
            raise ValueError(f"Signo inválido en restricciones: {sign}")
    return updated_obj

def standardize_restrictions(restrictions):
    updated_restrictions = []
    slack_count = 1
    artificial_count = 1
    for restr in restrictions:
        restr_copy = restr.copy()
        sign = restr_copy["sign"]
        if sign == "<=":
            slack_var = f"S{slack_count}"
            restr_copy[slack_var] = Fraction(1)
            slack_count += 1
        elif sign == ">=":
            slack_var = f"S{slack_count}"
            artificial_var = f"A{artificial_count}"
            restr_copy[slack_var] = Fraction(-1)
            restr_copy[artificial_var] = Fraction(1)
            slack_count += 1
            artificial_count += 1
        elif sign == "=":
            artificial_var = f"A{artificial_count}"
            restr_copy[artificial_var] = Fraction(1)
            artificial_count += 1
        else:
            raise ValueError(f"Signo inválido en restricciones: {sign}")
        restr_copy["sign"] = "="
        updated_restrictions.append(restr_copy)
    return updated_restrictions

def initialize_simplex_table(obj_func, restrictions):
    variables = list(obj_func.keys())
    table = []
    for i, restr in enumerate(restrictions):
        row = {}
        basic_var = None
        # Priorizar variables artificiales como básicas
        for var in variables:
            if var.startswith('A') and restr.get(var, Fraction(0)) == 1 and all(
                restrictions[r].get(var, Fraction(0)) == 0 for r in range(len(restrictions)) if r != i):
                basic_var = var
                break
        # Si no hay artificial, buscar cualquier básica
        if basic_var is None:
            for var in variables:
                if restr.get(var, Fraction(0)) == 1 and all(
                    restrictions[r].get(var, Fraction(0)) == 0 for r in range(len(restrictions)) if r != i):
                    basic_var = var
                    break
        row['Vb'] = basic_var
        row['Ci'] = obj_func.get(basic_var, Fraction(0))
        row['Bi'] = restr['value']
        for var in variables:
            row[var] = restr.get(var, Fraction(0))
        row['Øi'] = None
        table.append(row)
    return table, variables

def calculate_zj(table, variables):
    zj = {var: Fraction(0) for var in variables}
    zj['Bi'] = Fraction(0)
    for row in table:
        Ci = row['Ci']
        Bi = row['Bi']
        zj['Bi'] += Ci * Bi
        for var in variables:
            zj[var] += Ci * row.get(var, Fraction(0))
    return zj

def calculate_cj_minus_zj(obj_func, zj):
    result = {}
    for var in zj:
        if var == 'Bi':
            continue
        result[var] = obj_func.get(var, Fraction(0)) - zj.get(var, Fraction(0))
    return result

def find_pivot_column(cj_zj, optimization='max'):
    if optimization == 'max':
        max_val = max(cj_zj.values())
        if max_val <= 0:
            return None
        for var, val in cj_zj.items():
            if val == max_val:
                return var
    else:
        min_val = min(cj_zj.values())
        if min_val >= 0:
            return None
        for var, val in cj_zj.items():
            if val == min_val:
                return var
    return None

def find_pivot_row(table, pivot_var):
    ratios = []
    for i, row in enumerate(table):
        coef = row.get(pivot_var, Fraction(0))
        if coef > 0:
            ratio = row['Bi'] / coef
            ratios.append((ratio, i))
    if not ratios:
        return None
    _, row_index = min(ratios, key=lambda x: x[0])
    return row_index

def update_Øi(table, pivot_col):
    for row in table:
        coef = row.get(pivot_col, None)
        if coef is None or coef <= 0:
            row['Øi'] = None
        else:
            row['Øi'] = row['Bi'] / coef

def pivot_operation(table, pivot_row_idx, pivot_col_var, variables, obj_func):
    pivot_row = table[pivot_row_idx]
    pivot_element = pivot_row[pivot_col_var]

    print(f"\n--- Operación Pivote Iteración ---")
    print(f"Fila pivote: {pivot_row_idx + 1}, Variable pivote: {pivot_col_var}")
    print(f"Elemento pivote: {pivot_element}")
    print(f"Fila pivote antes de normalizar:")
    for var in variables + ['Bi']:
        print(f"  {var}: {pivot_row.get(var)}")

    # Normalizar fila pivote dividiendo todos los coeficientes por el pivote
    for var in variables + ['Bi']:
        pivot_row[var] = pivot_row.get(var, Fraction(0)) / pivot_element
    pivot_row['Ci'] = obj_func.get(pivot_col_var, Fraction(0))
    pivot_row['Vb'] = pivot_col_var

    print(f"Fila pivote después de normalizar:")
    for var in variables + ['Bi']:
        print(f"  {var}: {pivot_row.get(var)}")

    # Actualizar las otras filas
    for i, row in enumerate(table):
        if i == pivot_row_idx:
            continue
        factor = row.get(pivot_col_var, Fraction(0))
        print(f"\nActualizando fila {i + 1} con factor: {factor}")
        for var in variables + ['Bi']:
            old_val = row.get(var, Fraction(0))
            new_val = old_val - factor * pivot_row.get(var, Fraction(0))
            print(f"  {var}: {old_val} - {factor} * {pivot_row.get(var, Fraction(0))} = {new_val}")
            row[var] = new_val

def simplex_solver(optimization, objective, restrictions, M=1000):
    obj_func = parse_fraction_dict(objective)
    restrs = [parse_fraction_dict(r) for r in restrictions]
    std_obj_func = standardize_objective_function(obj_func, restrs, optimization, M)
    std_restrictions = standardize_restrictions(restrs)
    table, variables = initialize_simplex_table(std_obj_func, std_restrictions)
    iterations = []
    iteration = 0
    while True:
        iteration += 1
        zj = calculate_zj(table, variables)
        cj_zj = calculate_cj_minus_zj(std_obj_func, zj)
        pivot_col = find_pivot_column(cj_zj, optimization)
        if pivot_col is not None:
            update_Øi(table, pivot_col)
        else:
            for row in table:
                row['Øi'] = None
        iteration_data = {
            'iteration': iteration,
            'pivot_column': pivot_col,
            'table': [table_row_to_dict(row, variables) for row in table],
            'zj': fraction_to_str(zj),
            'cj_minus_zj': fraction_to_str(cj_zj)
        }
        iterations.append(iteration_data)
        if pivot_col is None:
            sol = {var: Fraction(0) for var in variables}
            for row in table:
                if row['Vb'] in variables:
                    sol[row['Vb']] = row['Bi']
            result = {
                'solution': {k: str(v) for k, v in sol.items()},
                'optimal_value': str(zj['Bi']),
            }
            break
        pivot_row_idx = find_pivot_row(table, pivot_col)
        if pivot_row_idx is None:
            result = {
                'error': 'Problema no acotado',
            }
            break
        pivot_operation(table, pivot_row_idx, pivot_col, variables, std_obj_func)
    return {
        'iterations': iterations,
        'result': result
    }

class SimplexInput(BaseModel):
    optimization: Literal['max', 'min']
    objective: Dict[str, str] = Field(..., example={"X1": "3/1", "X2": "5/1"})
    restrictions: List[Dict[str, str]] = Field(..., example=[
        {"X1": "1/1", "X2": "0/1", "sign": "<=", "value": "4/1"},
        {"X1": "0/1", "X2": "2/1", "sign": "<=", "value": "12/1"},
        {"X1": "3/1", "X2": "2/1", "sign": "<=", "value": "18/1"}
    ])
    M: int = Field(..., example=1000)

@app.post("/simplex")
def solve_simplex(data: SimplexInput):
    return simplex_solver(data.optimization, data.objective, data.restrictions, data.M)

@app.get("/")
async def root():
    return {"message": "CORS habilitado y funcionando"}