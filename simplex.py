# Aluno: Pedro Victor Soares da Silva Araújo
# O Programa consegue fazer a resolução dos PPLs testados, porem irei deixar o input comentado para facilitar o teste, para que não precise ficar inserindo valor a valor, testei com o PPL que o professor disponibilizou na página da atividade no class, e também testei com as duas atividades propostas no slide de Simplex Revisado.

import numpy as np

def take_values():
    
    """ 
    print("Método Simplex Revisado")
    num_var = int(input("Número de variáveis na função objetivo: "))
    num_rest = int(input("Número de restrições: "))

    print("\nDigite os coeficientes da função objetivo:")
    c = np.array([float(input(f"Coeficiente de x{i+1}: ")) for i in range(num_var)])

    A = []
    b = []
    operadores = []

    print("\nDigite os dados das restrições:")
    for i in range(num_rest):
        linha = [float(input(f"Coeficiente de x{j+1} na restrição {i+1}: ")) for j in range(num_var)]
        operador = input("Operador (<=, >=, =): ")
        valor_b = float(input("Valor do lado direito da restrição: "))
        A.append(linha)
        operadores.append(operador)
        b.append(valor_b)

    A = np.array(A)
    b = np.array(b)
    """
    
    # Coeficientes da função objetivo
    c = np.array([2, 3, 4])  
    # Restrições
    A = np.array([
        [1, 2, -3],  
        [-2, 0, 3],
        [1, 1, 0]
    ])
    b = np.array([10, 15, 8])  # Lado direito das restrições
    operadores = ["<=", ">=", "="]

    objetivo = True

    return A, b, c, operadores, objetivo

def forma_padrao(num_restricoes, num_variaveis):
    # Obter os valores iniciais
    matriz_A, matriz_b, matriz_c, operadores, objetivo = take_values()
    
    # Ajustar sinais para garantir que todos os valores de matriz_b sejam não negativos
    for i in range(num_restricoes):
        if matriz_b[i] < 0:
            
            matriz_A[i, :]  *= -1 # Inverter sinais da linha correspondente em A - uma linha (todas as colunas)
            matriz_b[i] *= -1 # Inverter o sinal em b
            
            if operadores[i] == '>=':
                operadores[i] = '<='
            else:
                operadores[i] = '>='

    # Contar variáveis adicionais
    variaveis_folga = sum(1 for op in operadores if op == '<=')
    variaveis_excedentes = sum(1 for op in operadores if op == '>=')
    variaveis_artificiais = sum(1 for op in operadores if op in ('=', '>='))
    total_variaveis = num_variaveis + variaveis_folga + variaveis_excedentes + variaveis_artificiais

    # Criar as novas matrizes
    A_padronizada = np.zeros((num_restricoes, total_variaveis))
    c_padronizada = np.zeros((1, total_variaveis))
    
    # Se o objetivo for maximizar, inverte os sinais de c
    if objetivo:  
        c_padronizada[0, :num_variaveis] = -matriz_c
    else:  
        c_padronizada[0, :num_variaveis] = matriz_c
    
    print(f"num_variaveis={num_variaveis}, variaveis_folga={variaveis_folga}, variaveis_excesso={variaveis_excedentes},, variaveis_artificiais={variaveis_artificiais}, total_variaveis={total_variaveis}")

    # Preencher as matrizes A e c
    offset = num_variaveis  # Começar após as variáveis originais
    for i in range(num_restricoes):
        A_padronizada[i, :num_variaveis] = matriz_A[i, :]  # Copiar as variáveis originais
        
        if operadores[i] == '<=':
            # Adicionar variável de folga
            A_padronizada[i, offset] = 1
            offset += 1
        elif operadores[i] == '>=':
            # Adicionar variável de excedência
            A_padronizada[i, offset] = -1
            offset += 1
            # Adicionar variável artificial
            A_padronizada[i, offset] = 1
            c_padronizada[0, offset] = abs(matriz_c[np.argmax(abs(matriz_c))])  # Custo alto
            offset += 1
        elif operadores[i] == '=':
            # Adicionar variável artificial
            A_padronizada[i, offset] = 1
            c_padronizada[0, offset] = abs(matriz_c[np.argmax(abs(matriz_c))])  # Custo alto
            offset += 1

    print("============================================")
    print("Matriz A (padronizada):\n", A_padronizada)
    print("Matriz b:\n", matriz_b)
    print("Matriz c (padronizada):\n", c_padronizada)
    print("============================================")

    return A_padronizada, matriz_b, c_padronizada

def simplex_revisado(custo, matriz_a, vetor_b):

    linhas, colunas = matriz_a.shape
    # Inicializando índices das variáveis básicas e não básicas
    indices_basicas = list(range(linhas))  # Variáveis básicas
    indices_nao_basicas = list(range(linhas, colunas))  # Variáveis não básicas

    # Matrizes e vetores iniciais
    matriz_basica = matriz_a[:, indices_basicas] # Matriz R
    matriz_nao_basica = matriz_a[:, indices_nao_basicas] # Matriz B
    matriz_cb = custo[0, indices_basicas] # Valores de c básicas
    matriz_cr = custo[0, indices_nao_basicas] # Valores de c não básicas
    valores_nao_basicos = np.zeros(len(indices_nao_basicas))  # Valores das variáveis não básicas inicialmente zero

    while True:
        # Custo reduzido
        inversa_matriz_basica = np.linalg.inv(matriz_basica)  # Inversa de B
        c_novos = matriz_cr - matriz_cb @ (inversa_matriz_basica @ matriz_nao_basica) # calculo de custo reduzido, que indica quanto a função objetivo melhora ou piora ao introduzir uma variável não básica na base.

        # @ - Multiplicação Matricial
        
        valores_basicos = inversa_matriz_basica @ vetor_b  # Calcular solução para variáveis básicas

        # Verificar otimalidade
        if np.all(c_novos >= 0):
            # Solução ótima encontrada
            solucao = np.zeros(colunas)
            solucao[indices_basicas] = valores_basicos.flatten() # Transforma em uma Unica dimensão
            solucao[indices_nao_basicas] = valores_nao_basicos
            valor_fo = matriz_cb @ valores_basicos
            return solucao, valor_fo

        # Determinar variável que entra na base
        indice_entra = np.argmin(c_novos) # Encontra o menor valor em c
        coluna_entra = matriz_nao_basica[:, indice_entra]

        # Determinar variável que sai da base
        direcao = inversa_matriz_basica @ coluna_entra
        if np.all(direcao <= 0):
            raise ValueError("Problema ilimitado.")
        
        razoes = valores_basicos / direcao
        razoes = np.where(direcao > 0, razoes, np.inf) # Para fazer substituição condicional
        indice_sai = np.argmin(razoes) # Encontra o indice que sai, ou seja, o menor

        # Atualizar bases
        variavel_entra = indices_nao_basicas[indice_entra]
        variavel_sai = indices_basicas[indice_sai]

        indices_basicas[indice_sai] = variavel_entra
        indices_nao_basicas[indice_entra] = variavel_sai
        matriz_basica = matriz_a[:, indices_basicas]
        matriz_nao_basica = matriz_a[:, indices_nao_basicas]
        matriz_cb = custo[0, indices_basicas]
        matriz_cr = custo[0, indices_nao_basicas]

if __name__ == "__main__":  

    print("\n           MÉTODO SIMPLEX REVISADO           \n")
    
    print("Balanceando o problema...")
    A_balanceada, artificiais, matriz_c = forma_padrao(3, 3)

    print("\nExecutando o método Simplex Revisado...")
    try:
        solucao, valor_objetivo = simplex_revisado(matriz_c, A_balanceada, artificiais)
        print("============================================")
        print("Solução ótima:", solucao)
        print("Valor ótimo da função objetivo:", valor_objetivo)
        print("============================================")
        
    except ValueError as e:
        print("\nErro:", e)