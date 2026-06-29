<p align="center">
  <strong>Concurrent and Distributed Programming - Heat Diffusion Simulation with CUDA and OpenMP</strong>
</p>

This repository contains the implementation of a **2D heat diffusion simulation** developed for the Concurrent and Distributed Programming course at the Federal University of São Paulo. The project compares the performance of three execution approaches: single-thread, multi-thread with OpenMP, and GPU parallelization with CUDA, measuring speedup and efficiency metrics across different thread configurations. Results are exported in CSV format for subsequent analysis.

## Objective

- Implement the 2D heat diffusion equation and solve it numerically using the finite difference method.
- Compare the performance between serial, parallel (OpenMP), and GPU (CUDA) execution.
- Measure speedup and efficiency for different thread counts.
- Export performance data for analysis and chart generation.

## Technologies Used

- **Languages:** C++ and CUDA C
- **CPU Parallelism:** OpenMP
- **GPU Parallelism:** NVIDIA CUDA
- **Compiler:** `nvcc` (NVIDIA CUDA Compiler)
- **Visual Interface:** ncurses (terminal mode)
- **Build System:** GNU Make
- **Platform:** Linux with an NVIDIA GPU

## Project Structure

```bash
PCD/
├── main.cu           # Entry point — orchestrates simulations and exports CSV
├── kernels.cu / .cuh # CUDA Kernels for GPU execution
├── simulations.cu / .cuh # Simulation functions (base, OpenMP, and CUDA)
├── matrix.cu / .cuh  # Data structures for 2D matrices
├── globals.cuh       # Global definitions and macros
├── outputwin.cu / .cuh # Output window with ncurses
├── makefile          # Project build file
├── run.sh            # Script to run batch experiments
├── filter.txt        # Filter for file synchronization (rclone)
├── tables/           # Generated results tables
├── vids/             # Simulation videos or animations
└── object/           # Compiled objects
```

## Source Code

### `main.cu`

The program's entry point. It receives command-line parameters (matrix dimensions, thread count, iterations, diffusion coefficient, time step, and delta_x) or uses default values. It runs the serial, OpenMP, and CUDA simulations, exporting the results in CSV format to stdout.

### `kernels.cu / kernels.cuh`

Contains the **CUDA kernels** that compute heat diffusion on the GPU. Each GPU thread processes a matrix cell independently.

### `simulations.cu / simulations.cuh`

Implements the three simulation modalities:
- **`simulate_base`:** Serial execution (single thread).
- **`simulate_OMP`:** Parallel execution with OpenMP (CPU multi-threading).
- **`simulate_cuda`:** GPU execution via CUDA kernels.

### `matrix.cu / matrix.cuh`

Defines the `float_2D` structure for representing two-dimensional floating-point matrices, including dynamic memory allocation and initialization macros.

## How to Run the Project

```bash
git clone https://github.com/g-reale/PCD.git
cd PCD

# Compile the project
make

# Run with default parameters (1000x1000 grid, 8 threads, 50 iterations)
./main

# Run with custom parameters
./main 0 1000 1000 8 50 0.1 0.01 1.0
```

**Command-line parameters:**

| Position | Parameter | Description |
|----------|-----------|-------------|
| 2 | M | Number of grid rows |
| 3 | N | Number of grid columns |
| 4 | threads | Number of threads |
| 5 | iterations | Number of iterations |
| 6 | diffusion | Diffusion coefficient |
| 7 | time_step | Time step |
| 8 | delta_x | Grid spacing |

### Batch Execution (Complete Experiment)

The `run.sh` script runs experiments for 5 thread configurations (2, 4, 8, 16, 32), saving results to CSV files:

```bash
./run.sh
```

Generates the following files: `data_2.csv`, `data_4.csv`, `data_8.csv`, `data_16.csv`, `data_32.csv`.

## Environment Setup

- **NVIDIA CUDA Toolkit** (with `nvcc`)
- CUDA-compatible **NVIDIA GPU**
- **OpenMP** (typically included with GCC/g++)
- **ncurses:** `sudo apt-get install libncurses-dev`
- **GNU Make**

```bash
# Verify nvcc
nvcc --version

# Compile
make
```

## Data Output (CSV)

The output is generated in CSV format on `stdout` with the following fields:

```
iteration; exec time (1 thread); exec time (N threads); exec time (N linear);
speedup; speedup (linear); efficiency; efficiency (linear);
difference (1 thread); difference (N threads)
```

The user can redirect this output to a file:

```bash
./main 0 5000 5000 8 500 0.1 0.01 1.0 > resultados.csv
```

## Workflow

1. A 2D grid is initialized with high temperature in the center and lower values surrounding it.
2. At each iteration, the finite difference method updates the temperature of each cell based on its neighbors.
3. The three implementations (serial, OpenMP, CUDA) run in parallel on the same initial grid.
4. Time, speedup, and efficiency metrics are calculated and exported per iteration.
5. The difference between serial and CUDA version results is monitored to verify numerical correctness.

---

<p align="center">
  <strong>Programação Concorrente e Distribuída - Simulação de Difusão de Calor com CUDA e OpenMP</strong>
</p>

Este repositório contém a implementação de uma **simulação de difusão de calor 2D** desenvolvida para a disciplina de Programação Concorrente e Distribuída realizada na Universidade Federal de São Paulo. O projeto compara o desempenho de três abordagens de execução,, single thread, multi-thread com OpenMP e paralelização em GPU com CUDA, medindo métricas de speedup e eficiência para diferentes configurações de threads. Os resultados são exportados em formato CSV para análise posterior.

## Objetivo

- Implementar a equação de difusão de calor em 2D e resolvê-la numericamente pelo método de diferenças finitas.
- Comparar o desempenho entre execução serial, paralela (OpenMP) e GPU (CUDA).
- Medir speedup e eficiência para diferentes números de threads.
- Exportar dados de performance para análise e geração de gráficos.

## Tecnologias Utilizadas

- **Linguagem:** C++ e CUDA C
- **Paralelismo CPU:** OpenMP
- **Paralelismo GPU:** NVIDIA CUDA
- **Compilador:** `nvcc` (NVIDIA CUDA Compiler)
- **Interface visual:** ncurses (modo terminal)
- **Build system:** GNU Make
- **Plataforma:** Linux com GPU NVIDIA

## Estrutura do Projeto

```bash
PCD/
├── main.cu           # Ponto de entrada — orquestra as simulações e exporta CSV
├── kernels.cu / .cuh # Kernels CUDA para execução na GPU
├── simulations.cu / .cuh # Funções de simulação (base, OpenMP e CUDA)
├── matrix.cu / .cuh  # Estruturas de dados para matrizes 2D
├── globals.cuh       # Definições globais e macros
├── outputwin.cu / .cuh # Janela de saída com ncurses
├── makefile          # Build do projeto
├── run.sh            # Script para executar experimentos em lote
├── filter.txt        # Filtro para sincronização de arquivos (rclone)
├── tables/           # Tabelas de resultados geradas
├── vids/             # Vídeos ou animações da simulação
└── object/           # Objetos compilados
```

## Códigos-fonte

### `main.cu`

Ponto de entrada do programa. Recebe parâmetros via linha de comando (dimensões da matriz, número de threads, iterações, coeficiente de difusão, passo de tempo e delta_x) ou usa valores padrão. Executa as simulações serial e CUDA em paralelo e imprime os resultados no formato CSV na saída padrão.

### `kernels.cu / kernels.cuh`

Contem os **kernels CUDA** que executam o cálculo da difusão de calor na GPU. Cada thread da GPU processa uma célula da matriz de forma independente.

### `simulations.cu / simulations.cuh`

Implementa as três modalidades de simulação:
- **`simulate_base`:** Execução serial (single thread).
- **`simulate_OMP`:** Execução paralela com OpenMP (multi-thread CPU).
- **`simulate_cuda`:** Execução na GPU via kernels CUDA.

### `matrix.cu / matrix.cuh`

Define a estrutura `float_2D` para representação de matrizes bidimensionais de ponto flutuante, com alocação dinâmica de memória e macros de inicialização.

## Como Executar o Projeto

```bash
git clone https://github.com/g-reale/PCD.git
cd PCD

# Compilar o projeto
make

# Executar com parâmetros padrão (grade 1000x1000, 8 threads, 50 iterações)
./main

# Executar com parâmetros personalizados
./main 0 1000 1000 8 50 0.1 0.01 1.0
```

**Parâmetros da linha de comando:**

| Posição | Parâmetro | Descrição |
|---------|-----------|-----------|
| 2 | M | Número de linhas da grade |
| 3 | N | Número de colunas da grade |
| 4 | threads | Número de threads |
| 5 | iterations | Número de iterações |
| 6 | diffusion | Coeficiente de difusão |
| 7 | time_step | Passo de tempo |
| 8 | delta_x | Espaçamento da grade |

### Execução em Lote (Experimento Completo)

O script `run.sh` executa experimentos para 5 configurações de threads (2, 4, 8, 16, 32), salvando os resultados em arquivos CSV:

```bash
./run.sh
```

Gera os arquivos: `data_2.csv`, `data_4.csv`, `data_8.csv`, `data_16.csv`, `data_32.csv`.

## Configuração do Ambiente

- **NVIDIA CUDA Toolkit** (com `nvcc`)
- **GPU NVIDIA** compatível com CUDA
- **OpenMP** (geralmente incluído no GCC/g++)
- **ncurses:** `sudo apt-get install libncurses-dev`
- **GNU Make**

```bash
# Verificar nvcc
nvcc --version

# Compilar
make
```

## Saída de Dados (CSV)

A saída é gerada no `stdout` em formato CSV com os seguintes campos:

```
iteration; exec time (1 thread); exec time (N threads); exec time (N linear);
speedup; speedup (linear); efficiency; efficiency (linear);
difference (1 thread); difference (N threads)
```

O usuário pode redirecionar a saída para um arquivo:

```bash
./main 0 5000 5000 8 500 0.1 0.01 1.0 > resultados.csv
```

## Fluxo de Funcionamento

1. Uma grade 2D é inicializada com um valor de temperatura elevado no centro e valores menores ao redor.
2. A cada iteração, o método de diferenças finitas atualiza a temperatura de cada célula com base nos seus vizinhos.
3. As três implementações (serial, OpenMP, CUDA) executam em paralelo para a mesma grade inicial.
4. Métricas de tempo, speedup e eficiência são calculadas e exportadas por iteração.
5. A diferença entre os resultados da versão serial e CUDA é monitorada para verificar a corretude numérica.