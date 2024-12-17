# **Vehicle Routing Problem with Time Windows (VRPTW)**

**Author**: [Camilo Bermúdez](https://www.github.com/camilobdez)

## Abstract

*This project addresses the Vehicle Routing Problem with Time Windows (VRPTW). The project implements and evaluates multiple heuristic and metaheuristic algorithms, including Constructive Heuristics, GRASP, Parallel Simulated Annealing, Parallel Variable Neighborhood Descent (VND), and a Genetic Algorithm (GA). Each method balances exploration and exploitation of the solution space to deliver high-quality, feasible routes. The algorithms are tested on benchmark instances derived from Solomon's VRPTW dataset, ranging from small instances with 25 customers to large-scale instances with 100 customers. Results demonstrate that the Parallel Simulated Annealing and the Genetic Algorithm achieve the most competitive solutions when prioritizing runtime and vehicle minimization, while the Parallel VND works better at reducing total distance at a higher computational cost. This work provides an organized codebase for testing various VRPTW heuristics, alongside detailed results, parameter sensitivity analysis, and runtime comparisons. Additionally, a presentation in Spanish offers further insights into the implementation and methodology of the algorithms.*

## **Index**

1. [Project Description](#project-description)
2. [Problem Definition](#problem-definition)
3. [Algorithms](#algorithms)
   - [Constructive Algorithm](#constructive-algorithm)
   - [Constructive Algorithm with Noise](#constructive-algorithm-with-noise)
   - [GRASP](#grasp)
   - [Parallel Simulated Annealing](#parallel-simulated-annealing)
   - [Parallel Variable Neighborhood Descent](#parallel-variable-neighborhood-descent)
   - [Genetic Algorithm](#genetic-algorithm)
4. [Results](#results)
5. [Performance Analysis](#performance-analysis)
6. [Conclusions](#conclusions)
7. [Appendix](#appendix)
   - [Setup and Formats](#setup-and-formats)
   - [File structure](#file-structure)
---

## **Project Description**

This project implements several heuristic algorithms to solve the Vehicle Routing Problem with Time Windows (VRPTW). The VRPTW is a combinatorial optimization problem where a set of routes must be constructed for vehicles to deliver goods to customers, minimizing both the number of vehicles and the total travel distance while respecting vehicle capacity constraints and time window constraints for each customer. 

The VRTPW has a wide range of real-world applications, particularly in logistics, transportation, and delivery services. Such as optimizing routes for delivery vehicles to ensure timely deliveries while minimizing the fleet size and travel distances, scheduling collection vehicles to service households or businesses within specific time windows, designing bus or shuttle routes to adhere to passenger schedules and reduce vehicle usage, planning efficient routes for mail and parcel delivery within predefined time windows, or managing the delivery of goods to warehouses or retail locations while meeting time-sensitive demands.

The implemented algorithms include:

1. **Constructive Algorithm**: A deterministic greedy method for constructing routes.
2. **Constructive Algorithm with Noise**: Adds controlled randomness to edge weights to diversify solutions.
3. **GRASP (Greedy Randomized Adaptive Search Procedure)**: Introduces probabilistic selection of edges during route construction.
4. **Parallel Simulated Annealing**: A metaheuristic optimization method that explores solutions using parallel simulated annealing processes.
5. **Parallel Variable Neighborhood Descent**: A local search method that systematically explores different neighborhoods to improve solutions.
6. **Genetic Algorithm**: An evolutionary optimization method that applies selection, crossover, and mutation to evolve high-quality solutions.

Each algorithm is evaluated on its ability to minimize both the number of vehicles (primary metric) and the total distance while providing diverse and feasible solutions.

---

## **Problem Definition**

In the **Vehicle Routing Problem with Time Windows (VRPTW)**:

- **Vehicles**:
   - Start and end at a depot (node $0$).
   - Have a maximum capacity $Q$ that cannot be exceeded.
- **Customer Nodes**:
   - Each node $i$ has a demand $q_i$.
   - Each node $i$ has a time window $[e_i, l_i]$, where $e_i$ is the earliest start time and $l_i$ is the latest time to begin service.
   - Each node $i$ has a service time $s_i$ required to unload the goods.
   - Distance between nodes $i$ and $j$ ($d_{ij}$).
   - $\text{MaxDistance}_0$: The maximum distance from the depot to any node.
- **Objective**:
   - Minimize the number of vehicles used.
   - Minimize the total travel distance, while ensuring all customers are serviced within their respective time windows.
     
---

## **Algorithms**

### **Constructive Algorithm**

#### **Graph Representation**

For some of the algorithms, the problem is represented as a graph $G = (V, E)$, where nodes represent the depot ($0$) and customer locations ($1, 2, \dots, n$) and the edges represent possible paths between nodes $i$ and $j$, subject to feasibility conditions (time windows and vehicle capacity). Each edge $(i, j)$ has a weight determined by the Euclidean distance $d_{ij}$ between nodes $i$ and $j$, the time window considerations at node $j$ and, in some cases, the proximity to the depot, adjusted using a bias parameter $\mu_{i,j}$.

#### **Graph Construction**
The graph is built with edge weights defined based on the following formula:

```math
\text{Weight}_{i,j} = \begin{cases}
\beta \cdot d_{ij} - (1 - \beta) \cdot l_j + \mu_{i,j}, & \text{if } i = 0 \\
\alpha \cdot d_{ij} + (1 - \alpha) \cdot e_j + \mu_{i,j}, & \text{if } i \neq 0
\end{cases}
```

Before adding an edge between nodes $i$ and $j$, the following feasibility conditions must be satisfied:

1. The vehicle must arrive at node $j$ before its time window closes: $\max(d_{0i}, e_i) + s_i + d_{ij} \leq l_j$.
2. The vehicle must return to the depot within its closing time window: $\max(d_{0i}, e_i) + s_i + d_{ij} + s_j + d_{j0} \leq l_0$.
3. The sum of the demands of nodes $i$ and $j$ must not exceed the vehicle capacity: $q_i + q_j \leq Q$.

#### **Purpose of Parameters**
- **$\alpha$**: Balances distance and time window readiness for non-depot nodes.
- **$\beta$**: Adjusts the importance of servicing distant nodes early versus nodes with earlier time window closures. It is applied only to edges from the depot ($i = 0$). That way ensures that initial routes from the depot prioritize distant nodes or nodes with tight time windows, improving overall route structure.
- **$\mu_{i,j}$**: Adds bias to edge weights based on relative proximity to the depot, so it adjusts edge weights to favor nodes that are farther from the depot and helps in constructing efficient routes for distant areas:

```math
\mu_{i,j} = \left( \frac{d_{0i}}{\text{MaxDistance}_0} \cdot \frac{d_{0j}}{\text{MaxDistance}_0} \right) \cdot \left( d_{ij} - d_{j0} \right)
```


#### **Route Construction**
1. Start at the depot (node $0$).
2. At each step, select the best feasible neighbor based on sorted edge weights.
3. Continue until all nodes are visited while respecting:
   - Vehicle capacity constraints.
   - Time window constraints.
4. Return to the depot to complete the route.

---

### **Constructive Algorithm with Noise**

#### **Graph Construction**
The graph is constructed similarly to the Constructive Algorithm, but noise is added to the edge weights to introduce variability:

```math
\text{Weight}_{i,j} = \begin{cases}
\beta \cdot d_{ij} - (1 - \beta) \cdot l_j + r, & \text{if } i = 0 \\
\alpha \cdot d_{ij} + (1 - \alpha) \cdot e_j + r, & \text{if } i \neq 0
\end{cases}
```

#### **Noise Generation**
Noise is sampled from a **uniform distribution** based on $\mu_{i,j}$:

- If $\mu_{i,j} > 0$: $\text{r} \sim U(0, \mu_{i,j})$
- If $\mu_{i,j} < 0$: $\text{r} \sim U(\mu_{i,j}, 0)$
- If $\mu_{i,j} = 0$, no noise is added.

#### **Route Construction**
1. The routes are built using the same process as in the Constructive Algorithm.
2. The noise introduces variability, leading to different solutions across runs.

---

### **GRASP**

#### **Graph Construction**
The graph is constructed using the same weight formula as in the Constructive Algorithm, including the role of **$\mu_{i,j}$** and **$\beta$** to adjust edge weights.

#### **Route Construction**
1. Start at the depot (node $0$).
2. Build a list of feasible neighbors based on constraints:
   - Time window constraints.
   - Vehicle capacity.
3. Compute edge weights for the neighbors, influenced by $\alpha$, $\beta$, and $\mu_{i,j}$.
4. Select a neighbor probabilistically using a geometric distribution: $P(k) = (1 - p)^{k - 1} p, \quad k \geq 1$, where $k$ is the index of the selected neighbor in the sorted list.
5. If the generated index exceeds the candidate list size, repeat the selection.
6. Continue until all nodes are visited and return to the depot.

The geometric distribution skews the selection towards better candidates (lower $k$ values) while still allowing exploration of alternative options.

---

### **Parallel Simulated Annealing**

The Parallel Simulated Annealing (Parallel SA) algorithm extends the classical Simulated Annealing by allowing multiple threads to explore solutions independently while periodically exchanging solutions to enhance the overall search process.

Each thread starts with an initial solution generated using the Constructive Algorithm with Noise. The temperature for each thread is initialized as $T_0 = \gamma \cdot \text{cost}(\text{initial solution})$, where $\gamma$ is a temperature scaling factor. The temperature controls the probability of accepting worse solutions during the search.

Each thread explores the solution space using Simulated Annealing. At each iteration:

1. A new solution is generated by perturbing the current solution using one of the following neighborhood operators:
   - Shift: Move a node to a different position or route.
   - Rearrange: Reorder nodes within the same route.
   - Exchange: Swap nodes or segments between two routes.
   - Reverse: Reverse the order of a segment within a route.
     
2. The cost of the new solution is evaluated. If it improves the current solution in the thread, it is accepted. If the new solution is worse, it is accepted with a probability $\text{P} = \frac{T + \delta}{T}$, where $T$ is the current temperature and $\delta$ is the cost difference between the new solution and the current solution. A random number between 0 and 1 determines whether the solution is accepted.

3. The temperature is gradually decreased ($T_{i+1} = \beta T_i$) during the search to reduce the acceptance of worse solutions and focus on exploitation.

Every $\omega$ iterations, threads cooperate by exchanging their local best solutions sequentially. Thread 1 remains independent and does not receive solutions from other threads. And for each thread $t$ ($t > 1$), the solution from its left neighbor (thread $t-1$) is evaluated. If the left neighbor's solution has a better cost, it replaces the thread's current local best solution.

If the equilibrium counter reaches a threshold $\tau$ (indicating stagnation), the temperature is re-initialized: $T = \gamma \cdot \text{cost}(\text{current solution})$. This helps the algorithm escape local optima by temporarily increasing the acceptance probability for worse solutions. The algorithm terminates when all threads experience stagnation (no improvements for a prolonged period).

---

### **Parallel Variable Neighborhood Descent**

The Parallel Variable Neighborhood Descent algorithm extends the classical VND by distributing the search process across multiple threads, where each thread explores the solution space independently while systematically applying neighborhood structures.

Each thread starts with an initial solution generated using the Constructive Algorithm with Noise. Threads iteratively improve their solutions by exploring predefined neighborhoods in a structured order.

At each iteration:

1. A neighborhood operator is applied to perturb the current solution. The operators are systematically applied in the following order:
   - Shift: Move a node to a different position or route.
   - Rearrange: Reorder a sequence of nodes within a single route.
   - Exchange: Swap nodes or segments between two routes.
   - Swap: Swap single nodes between two routes.
   - Reverse: Reverse the order of a segment within a route.

2. The perturbation step is followed by a local improvement phase using the `first_local_search` function. This function refines the perturbed solution to ensure that local improvements are applied before moving to the next neighborhood.

3. If the new solution improves the current solution, the search restarts from the first neighborhood. If no improvement is found, the search progresses to the next neighborhood in the sequence.

4. Each thread explores its assigned neighborhoods independently. Threads keep track of their local best solutions during this process.

Periodically, the global best solution is updated based on the best local solutions found by all threads. This ensures that the search process benefits from the best solutions discovered across all threads.The algorithm terminates when all threads have explored their neighborhoods without finding further improvements.

---

### **Genetic Algorithm**

Each solution, referred to as a chromosome, is represented as a sequence of customer nodes, which is a permutation of all customers. Routes are not explicitly defined in the chromosome. Instead, during the decoding phase, valid routes are constructed starting from the depot, ensuring all constraints are satisfied. The initial population consists of a mix of random and greedy solutions. Random solutions are generated by permuting the list of customers, while greedy solutions are constructed by starting from a random customer and iteratively adding the nearest unvisited customer within a predefined radius.

To decode a chromosome into a feasible VRPTW solution, the following process is applied:
1. Customers are sequentially added to a route while checking vehicle capacity and time window feasibility
2. After all routes are constructed, the algorithm attempts to move the last customer of each route to the beginning of the next route. This reinsertion is performed only if it improves the solution by reducing the objective function (number of vehicles or total distance) and respecting all constraints.  

The fitness of each solution is evaluated based on the two objectives (number of vehicles and total distance). To handle these objectives simultaneously, Pareto Ranking is used. A solution is said to dominate another if it is no worse in all objectives and strictly better in at least one.

At each generation, the algorithm evolves the population through selection, crossover, and mutation.

The selection process is used to choose two parents for crossover. For each parent:
1. Four individuals are randomly selected from the population.  
2. Among these four, the best solution is identified based on its Pareto rank.  
3. With a probability r (where r > 0.5), the best individual is selected.  
4. If the best individual is not selected, one of the other three individuals is chosen randomly.  

This process ensures that solutions with better fitness have a higher chance of being selected, but diversity is maintained by occasionally selecting other individuals.

The crossover operator combines genetic material from two parents to create an offspring. It works as follows:
1. A random route (a sequence of customer nodes) is selected from Parent 1.  
2. The nodes in the selected route are removed from Parent 2 to prevent duplicate visits.  
3. The removed nodes are reinserted into the offspring at positions that minimize the objective function (number of vehicles and total distance). Nodes are reinserted one by one, ensuring that vehicle capacity and time window constraints are satisfied.

The offspring retains structural characteristics from both parents while ensuring feasibility and improving solution quality.

Mutation introduces diversity into the population by altering solutions slightly. For each individual, a mutation is applied with a predefined probability. The mutation involves reversing a segment of two or three consecutive nodes within a route. The mutation is only accepted if the resulting solution respects constraints and improves the solution.

At the end of each generation elitism is applied to preserve the best solutions (Pareto rank 1) in the population. This guarantees that the best solutions discovered so far are not lost, then, parents are selected using tournament selection, offspring are generated through crossover and mutated to introduce diversity, and finally the population is replaced with the newly generated individuals, maintaining the population size. The algorithm evolves the population for a predefined number of generations and the best solutions are identified as the non-dominated solutions in the final population. 

---

## **Results**

The instances used in this project come from the Solomon Benchmark for the Vehicle Routing Problem with Time Windows (VRPTW). Specifically:

- Instances VRPTW1, VRPTW7, and VRPTW13 correspond to the C101 instance from Solomon's benchmark, with 25, 50, and 100 customers, respectively.
- Instances VRPTW2, VRPTW8, and VRPTW14 correspond to the C201 instance with 25, 50, and 100 customers.
- Instances VRPTW3, VRPTW9, and VRPTW15 correspond to the R101 instance with 25, 50, and 100 customers.
- Instances VRPTW4, VRPTW10, and VRPTW16 correspond to the R201 instance with 25, 50, and 100 customers.
- Instances VRPTW5, VRPTW11, and VRPTW17 correspond to the RC101 instance with 25, 50, and 100 customers.
- Instances VRPTW6, VRPTW12, and VRPTW18 correspond to the RC201 instance with 25, 50, and 100 customers.

The last six instances (RPTW13 to VRPTW18) correspond to the original Solomon instances with 100 customers, which are available at **[Solomon Benchmark Instances](https://www.sintef.no/projectweb/top/vrptw/100-customers/)**.

These benchmark instances are widely used to evaluate VRPTW algorithms due to their standardized format and varying levels of complexity (clustered, random, and mixed customer distributions).

### **Number of Vehicles Comparison**

| **Instance** | **Constructive** | **Constructive with Noise** | **GRASP** | **Parallel SA** | **Parallel VND** | **GA** |
|--------------|------------------|-----------------------------|-----------|-----------------|------------------|--------|
| VRPTW1       | 3                | 3                           | 3         | 3               | 3                | 3      |
| VRPTW2       | 2                | 2                           | 3         | 2               | 2                | 2      |
| VRPTW3       | 8                | 8                           | 8         | 8               | 8                | 8      |
| VRPTW4       | 2                | 2                           | 2         | 2               | 2                | 2      |
| VRPTW5       | 4                | 4                           | 4         | 4               | 5                | 4      |
| VRPTW6       | 2                | 2                           | 2         | 2               | 2                | 2      |
| VRPTW7       | 5                | 5                           | 6         | 5               | 5                | 5      |
| VRPTW8       | 2                | 2                           | 2         | 2               | 2                | 2      |
| VRPTW9       | 12               | 12                          | 12        | 11              | 12               | 11     |
| VRPTW10      | 3                | 3                           | 3         | 2               | 3                | 2      |
| VRPTW11      | 9                | 8                           | 9         | 8               | 9                | 8      |
| VRPTW12      | 3                | 3                           | 3         | 3               | 3                | 3      |
| VRPTW13      | 10               | 10                          | 12        | 10              | 10               | 10     |
| VRPTW14      | 3                | 3                           | 4         | 3               | 3                | 3      |
| VRPTW15      | 21               | 20                          | 20        | 19              | 20               | 19     |
| VRPTW16      | 5                | 5                           | 4         | 4               | 4                | 4      |
| VRPTW17      | 17               | 16                          | 16        | 16              | 17               | 15     |
| VRPTW18      | 5                | 4                           | 5         | 4               | 4                | 4      |
| **Average**  | **6.444**        | **6.222**                   | **6.556** | **6.000**       | **6.333**        | **5.944** |

### **Total Distance Comparison**

| **Instance** | **Constructive** | **Constructive with Noise** | **GRASP** | **Parallel SA** | **Parallel VND** | **GA** |
|--------------|------------------|-----------------------------|-----------|-----------------|------------------|--------|
| VRPTW1       | 191.815          | 191.815                     | 191.815   | 191.815         | 191.815          | 191.815 |
| VRPTW2       | 215.542          | 215.542                     | 292.776   | 215.542         | 215.542          | 228.185 |
| VRPTW3       | 660.155          | 660.155                     | 630.379   | 618.328         | 618.328          | 624.633 |
| VRPTW4       | 719.561          | 647.436                     | 577.368   | 524.590         | 524.590          | 523.655 |
| VRPTW5       | 488.446          | 470.616                     | 470.951   | 462.153         | 476.953          | 462.153 |
| VRPTW6       | 572.386          | 531.343                     | 461.359   | 432.295         | 432.295          | 434.226 |
| VRPTW7       | 363.248          | 363.248                     | 440.698   | 363.248         | 363.248          | 363.248 |
| VRPTW8       | 497.174          | 497.174                     | 444.959   | 444.959         | 444.959          | 444.959 |
| VRPTW9       | 1133.41          | 1124.37                     | 1088.08   | 1100.72         | 1046.70          | 1108.86 |
| VRPTW10      | 1179.64          | 1150.46                     | 1065.58   | 1019.10         | 871.628          | 971.184 |
| VRPTW11      | 1059.97          | 988.526                     | 993.748   | 951.738         | 958.832          | 946.457 |
| VRPTW12      | 1333.70          | 1199.92                     | 1041.56   | 852.983         | 852.983          | 842.965 |
| VRPTW13      | 855.065          | 855.065                     | 1042.74   | 828.937         | 828.937          | 828.937 |
| VRPTW14      | 591.555          | 591.555                     | 622.868   | 591.555         | 591.555          | 591.555 |
| VRPTW15      | 1953.76          | 1815.71                     | 1880.76   | 1678.68         | 1654.63          | 1723.34 |
| VRPTW16      | 1599.72          | 1533.43                     | 1696.01   | 1337.87         | 1337.87          | 1312.40 |
| VRPTW17      | 1879.78          | 1876.06                     | 1805.32   | 1700.27         | 1691.02          | 1715.05 |
| VRPTW18      | 1707.24          | 1699.14                     | 1583.35   | 1504.79         | 1504.79          | 1534.71 |
| **Average**  | **944.565**      | **911.754**                 | **907.240** | **823.310**   | **811.482**      | **824.907** |


### **Comparison with Literature Results**


| **Instance** | **Best Vehicles (Literature)** | **Best Vehicles (Methods)** | **Best Distance (Literature)** | **Best Distance (Methods)** |
|--------------|--------------------------------|-----------------------------|-------------------------------|----------------------------|
| VRPTW13      | **10**                         | 10 (SA, VNS, GA)            | **828.94**                    | 828.94 (SA, VND, GA)       |
| VRPTW14      | **3**                          | 3 (All Methods)             | **591.56**                    | 591.55 (All Methods)       |
| VRPTW15      | **19**                         | 19 (SA, GA)                 | **1650.80**                   | 1654.63 (VND)              |
| VRPTW16      | **4**                          | 4 (SA, VNS, GA)             | **1252.37**                   | 1337.87 (SA, VND)          |
| VRPTW17      | **14**                         | 15 (GA)                     | **1696.95**                   | 1691.02 (VNS)              |
| VRPTW18      | **4**                          | 4 (SA, VNS, GA)             | **1406.94**                   | 1504.79 (SA, VND)          |

---

## **Performance Analysis**

The performance of the methods is evaluated based on two primary metrics: the number of vehicles (the primary objective) and the total travel distance, while also considering computational efficiency to understand scalability across different instance sizes.

### **Parameter Sensitivity**
The Constructive Heuristic with Noise demonstrated sensitivity to the parameters $\alpha$ and $\beta$, which control the balance between distance and time window readiness. For instances with tight time windows, lower values of $\alpha$ that emphasize time readiness improved the solutions. Conversely, spatially spread instances benefited from higher $\alpha$, where prioritizing distance resulted in superior outcomes.

In the Simulated Annealing, the initial temperature $T_0$ and the cooling rate $\beta$ played a significant role in determining performance. A higher initial temperature allowed for broader exploration of the solution space but increased runtime, while lower temperatures restricted exploration too early, leading to premature convergence. A cooling rate between 0.92 and 0.95 balanced exploration and exploitation effectively. Cooperation intervals between threads also impacted results; frequent exchanges improved convergence but introduced some overhead.

The Genetic Algorithm depended heavily on the tournament selection probability $r$ and the number of generations. When $r>0.5$, the algorithm favored exploitation by selecting high-quality solutions, whereas lower values promoted diversity, helping the algorithm avoid stagnation. Generally, improvements diminished after 200 generations, as solutions stabilized.

### **Algorithm Comparisons**

The Constructive Heuristic provided the fastest and simplest solutions but consistently produced the highest number of vehicles and the longest total distances. It serves as an effective baseline for comparison due to its speed and computational efficiency.

By introducing variability during route construction, the Constructive Heuristic with Noise improved upon the baseline method. It achieved better average vehicle counts than the Parallel VND while maintaining the same runtime efficiency as the deterministic heuristic.

GRASP, while offering a good trade-off between exploration and runtime, performed slightly worse in terms of vehicle counts compared to the Constructive Heuristic with Noise. Nevertheless, its runtime remained competitive with the constructives, making it a practical choice for quickly generating feasible solutions.

The Parallel Simulated Annealing (SA) consistently delivered high-quality solutions, achieving competitive results for both vehicle counts and total distances. It significantly outperformed GRASP and the Constructive methods in solution quality while maintaining reasonable runtimes. The cooperation mechanism between threads enhanced convergence without adding substantial overhead.

Parallel Variable Neighborhood Descent (VND) produced the shortest total distances among all methods but underperformed in terms of the primary metric—the number of vehicles. Despite its strong results for distance minimization, it was the slowest algorithm overall, making it less suitable when computational efficiency is a concern.

Finally, the Genetic Algorithm (GA) emerged as the fastest metaheuristic, balancing runtime and solution quality effectively. It delivered competitive results for both metrics, offering a scalable and efficient solution for large instances. The crossover operator played a significant role in maintaining solution quality, while tournament selection ensured effective exploration of the solution space.

---


## **Conclusions**

The Parallel Simulated Annealing (SA) and the Genetic Algorithm (GA) stood out as the most effective methods overall. Both achieved competitive results in minimizing the number of vehicles—the primary objective—while maintaining acceptable runtimes. Parallel SA consistently delivered high-quality solutions through its exploration-exploitation balance, while GA excelled in speed and scalability, making it a practical choice for large-scale instances.

Parallel Variable Neighborhood Descent (VND) demonstrated its strength in minimizing total travel distance, achieving the best results for this metric. However, it underperformed in the primary objective of minimizing vehicles and was computationally the slowest, making it less favorable when runtime is critical.

The Constructive Heuristics (deterministic and noise-enhanced) served as fast baseline methods, offering quick solutions with reasonable quality. The Constructive Heuristic with Noise, in particular, outperformed GRASP and Parallel VND in vehicle count while maintaining comparable runtimes to the basic heuristic.

Ultimately, the choice of method depends on the specific requirements of the problem. For scenarios where runtime is critical, the Genetic Algorithm or Constructive Heuristics are preferred. For applications prioritizing solution quality, especially total distance, Parallel SA and GA offer robust solutions at the cost of additional computational time.

---
## **Appendix**

### **Setup and Formats**

To set up and compile the project, ensure the following dependencies are installed:
- **C++ Compiler**: A standard C++ compiler (e.g., g++ or MSVC).
- **OpenMP**: Required for parallel execution of Simulated Annealing and Variable Neighborhood Descent.

To compile the code for Parallel SA and VND, run the following command:

```bash
 g++ -std=c++11 -fopenmp algorithm.cpp -o vrptw_solver
```

Replace `algorithm.cpp` with the respective source file.

The input file follows the standard Solomon VRPTW benchmark format:

```
<Number of Customers> <Vehicle Capacity>
<Node ID> <X-Coordinate> <Y-Coordinate> <Demand> <Ready Time> <Due Time> <Service Time>
...
```
The output includes the generated routes and their associated metrics as follows:

```
<Number of Vehicles> <Total Distance> <Computation Time in ms>
<Number of Nodes in Route 1 (excluding depots)> <Route Nodes> <Arrival Times> <Vehicle Load>
<Number of Nodes in Route 2 (excluding depots)> <Route Nodes> <Arrival Times> <Vehicle Load>
...
```

### **File structure**

The project is organized into the following folder structure:

```
project_root/
│
├── algorithms/
│   ├── 1-constructive.cpp
│   │── 2-constructive-with-noise.cpp
│   ├── 3-grasp.cpp
│   ├── 4-parallel_sa.cpp
│   ├── 5-parallel_vnd.cpp
│   └── 6-genetic-algorithm.cpp
│
├── results/
│   ├── 1-constructive.xlsx
│   ├── 2-constructive-with-noise.xlsx
│   └── ...
│
├── presentations/
│   ├── 1-constructivos.pdf
│   │── 2-metaheurísticos.pdf
│   └── 3-evolutivos.pdf
│
└── README.md
```

- **`algorithms/`**: Contains the C++ source files implementing the heuristic and metaheuristic algorithms.
- **`results/`**: Stores the respective output files for the test instances, including routes, number of vehicles, and total distances.
- **`presentations/`**: Includes a detailed PDF presentation (in Spanish) explaining the implemented methods, their logic, and the experimental setup.
- **`README.md`**: The current documentation file describing the project structure, setup, algorithms, and performance analysis.
