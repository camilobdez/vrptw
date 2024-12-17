#include <bits/stdc++.h>
#include <windows.h>
using namespace std;

#define sz(x) (int)x.size()
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define pb push_back

double d[101][101] = {0.0};
int n, Q, population_size = 300, generations = 10;

string get_current_directory() {
    char buffer[MAX_PATH];
    GetCurrentDirectory(MAX_PATH, buffer);
    return string(buffer);
}


struct Node {
    int index;
    double x, y;
    int q;
    double e, l;
    double s;
};

vector<Node> nodes;

struct Vehicle {
    int id;
    int load;
    vector<int> route;
    vector<double> arrivals;
    double traveled_distance;

    Vehicle() : id(-1), load(0), traveled_distance(0.0) {}

    bool update_times(int beginning = 1) {
        if (arrivals.size() != route.size()) {
            arrivals.resize(route.size());
            arrivals[0] = 0.0;  // depot start time
            beginning = 1;      // force full recalculation
        }

        for (int i = 1; i < sz(arrivals); i++) {
            const Node& prev_node = nodes[route[i - 1]];
            const Node& curr_node = nodes[route[i]];

            double current_time = arrivals[i - 1] + prev_node.s + d[route[i - 1]][route[i]];
            if (current_time > curr_node.l) return false;
            arrivals[i] = max(current_time, curr_node.e);
        }
        return true;
    }

    double update_load(int beginning = 0) {
        load = 0;
        for (int i = 0; i < sz(route); i++) {
            load += nodes[route[i]].q;
        }
        return load;
    }

    double update_distance(int beginning) {
        double acum_from = 0;
        for (int i = beginning; i < sz(route); i++)
            acum_from += d[route[i - 1]][route[i]];
        traveled_distance += acum_from;
        return acum_from;
    }
};

struct Solution {
    int num_vehicles;
    double total_distance;
    int computation_time;
    vector<Vehicle> vehicles;

    void calculate_total_distance() {
        total_distance = 0;
        for (auto v: vehicles)
            total_distance += v.traveled_distance;
    }
};

void read_instance(const string& file_path) {
    ifstream file(file_path);
    file >> n >> Q;
    nodes.resize(n + 1);

    for (int i = 0; i <= n; i++)
        file >> nodes[i].index >> nodes[i].x >> nodes[i].y >> nodes[i].q >> nodes[i].e >> nodes[i].l >> nodes[i].s;

    file.close();
}

void out_solution(Solution& sol) {
    cout << sol.num_vehicles << " " << sol.total_distance << " " << sol.computation_time;
    for (int i = 0 ; i<sol.num_vehicles; i++) {
        cout << "\n" << sz(sol.vehicles[i].route)-2 << " ";
        for (auto &j: sol.vehicles[i].route)
            cout << j << " ";
        for (auto &j: sol.vehicles[i].arrivals)
            cout << j << " ";
        cout << sol.vehicles[i].load;
    }
}


void compute_distances() {
    for (int i = 0; i <= n; i++) {
        for (int j = i + 1; j <= n; j++) {
            double dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
            d[i][j] = round(sqrt(dx * dx + dy * dy) * 1000) / 1000;
            d[j][i] = d[i][j];
        }
    }
}

struct Individual {
    vector<int> chromosome;  // sequence of customer visits
    double fitness_vehicles; // number of vehicles
    double fitness_distance; // total distance
    int rank;               // for Pareto ranking
};

vector<Individual> population;

// Check if adding a customer to a route is feasible
bool is_feasible_addition(Vehicle& vehicle, int customer_id) {
    // Add depot if this is first customer
    if (vehicle.route.empty()) {
        vehicle.route.push_back(0);  // depot
        vehicle.arrivals.push_back(0.0);  // start time at depot
    }

    // Check capacity constraint - create temporary calculation
    Vehicle temp = vehicle;
    temp.route.push_back(customer_id);
    temp.load = 0;

    if (temp.update_load(1) > Q) {  // Calculate full route load
        return false;
    }

    // Calculate arrival time at new customer
    double prev_node_time = vehicle.arrivals.back() + nodes[vehicle.route.back()].s;
    double travel_time = d[vehicle.route.back()][customer_id];
    double arrival_time = prev_node_time + travel_time;

    // Check time window constraint
    if (arrival_time > nodes[customer_id].l) {
        return false;
    }

    // Adjust for early arrival
    arrival_time = max(arrival_time, nodes[customer_id].e);

    // Check if can return to depot
    double return_time = arrival_time + nodes[customer_id].s + d[customer_id][0];
    if (return_time > nodes[0].l) {
        return false;
    }

    return true;
}
// Phase 1: Initial route construction
Solution construct_initial_routes(const vector<int>& chromosome) {
    Solution solution;
    Vehicle current_vehicle;
    current_vehicle.id = 0;
    
    for (int customer : chromosome) {
        bool customer_added = false;
        
        // Try to add to current vehicle
        if (is_feasible_addition(current_vehicle, customer)) {
            // Add customer to route
            current_vehicle.route.push_back(customer);
            
            // Update arrival time
            double prev_node_time = current_vehicle.arrivals.back() + 
                                  nodes[current_vehicle.route[current_vehicle.route.size()-2]].s;
            double travel_time = d[current_vehicle.route[current_vehicle.route.size()-2]][customer];
            double arrival_time = max(prev_node_time + travel_time, nodes[customer].e);
            current_vehicle.arrivals.push_back(arrival_time);
            
            // Update vehicle load and distance
            current_vehicle.update_load();  // Changed this line
            current_vehicle.traveled_distance += d[current_vehicle.route[current_vehicle.route.size()-2]][customer];
            
            customer_added = true;
        }
        
        // If customer couldn't be added, start new vehicle
        if (!customer_added) {
            // Add return to depot for current vehicle if not empty
            if (!current_vehicle.route.empty()) {
                current_vehicle.route.push_back(0);
                current_vehicle.traveled_distance += d[current_vehicle.route[current_vehicle.route.size()-2]][0];
                solution.vehicles.push_back(current_vehicle);
            }
            
            // Start new vehicle
            current_vehicle = Vehicle();
            current_vehicle.id = solution.vehicles.size();
            current_vehicle.route.push_back(0);  // start at depot
            current_vehicle.arrivals.push_back(0.0);
            
            // Add customer to new vehicle
            current_vehicle.route.push_back(customer);
            double arrival_time = max(d[0][customer], nodes[customer].e);
            current_vehicle.arrivals.push_back(arrival_time);
            current_vehicle.update_load();  // Changed this line
            current_vehicle.traveled_distance = d[0][customer];
        }
    }
    
    // Add final return to depot if needed
    if (!current_vehicle.route.empty()) {
        current_vehicle.route.push_back(0);
        current_vehicle.update_times();
        current_vehicle.traveled_distance += d[current_vehicle.route[current_vehicle.route.size()-2]][0];
        solution.vehicles.push_back(current_vehicle);
    }
    current_vehicle.update_times();
    
    return solution;
}
// Phase 2: Route improvement
void improve_routes(Solution& solution) {
    for (size_t i = 0; i < solution.vehicles.size() - 1; i++) {
        Vehicle& current_route = solution.vehicles[i];
        Vehicle& next_route = solution.vehicles[i+1];
        
        // Skip if route is too short
        if (current_route.route.size() <= 2) continue;  // needs at least depot-customer-depot
        if (next_route.route.size() < 2) continue;      // needs at least depot-depot

        // Get the last customer before the final depot
        int customer_to_move = current_route.route[current_route.route.size()-2];

        // Create new version of next route with the customer at start
        Vehicle temp_next = next_route;
        temp_next.route.clear();
        temp_next.arrivals.clear();
        temp_next.load = 0;
        temp_next.traveled_distance = 0;

        // Build new next route
        temp_next.route.push_back(0);              // start depot
        temp_next.route.push_back(customer_to_move);  // add moved customer
        // Add remaining customers from original next route (skip first and last depot)
        for(size_t j = 1; j < next_route.route.size()-1; j++) {
            temp_next.route.push_back(next_route.route[j]);
        }
        temp_next.route.push_back(0);  // end depot

        // Check if new route is feasible
        if (!temp_next.update_times(1)) continue;
        if (temp_next.update_load(1) > Q) continue;
        temp_next.update_distance(1);

        // Create new version of current route without the customer we're moving
        Vehicle temp_current = current_route;
        temp_current.route.clear();
        temp_current.arrivals.clear();
        temp_current.load = 0;
        temp_current.traveled_distance = 0;

        // Rebuild current route without the customer we're moving
        if (current_route.route.size() > 3) {  // if there are other customers besides the one we're moving
            temp_current.route.push_back(0);  // start depot
            for(size_t j = 1; j < current_route.route.size()-2; j++) {
                temp_current.route.push_back(current_route.route[j]);
            }
            temp_current.route.push_back(0);  // end depot
            
            if (!temp_current.update_times(1)) continue;
            temp_current.update_load(1);
            temp_current.update_distance(1);
        }

        // Calculate total distance - if current route became empty, only count next route
        double old_distance = current_route.traveled_distance + next_route.traveled_distance;
        double new_distance = (temp_current.route.size() > 2 ? temp_current.traveled_distance : 0) + 
                            temp_next.traveled_distance;

        // If improvement found, update routes
        if (new_distance < old_distance) {
            // If current route became empty (only had one customer that we moved),
            // we'll mark it for removal by making it empty
            if (temp_current.route.size() <= 2) {
                temp_current.route.clear();
                temp_current.arrivals.clear();
                temp_current.load = 0;
                temp_current.traveled_distance = 0;
            }
            solution.vehicles[i] = temp_current;
            solution.vehicles[i+1] = temp_next;
        }
    }

    // Remove empty routes
    vector<Vehicle> valid_vehicles;
    for (Vehicle& v : solution.vehicles) {
        if (!v.route.empty()) {  // Keep only non-empty routes
            v.update_times();
            valid_vehicles.push_back(v);
        }
    }
    solution.vehicles = valid_vehicles;
}
// Main decoding function
Solution decode_chromosome(const vector<int>& chromosome) {
    // Phase 1: Construct initial routes
    Solution solution = construct_initial_routes(chromosome);
    
    // Phase 2: Improve routes
    improve_routes(solution);
    
    // Update solution metrics  
    solution.num_vehicles = solution.vehicles.size();
    solution.calculate_total_distance();
    
    return solution;
}

// Generate random chromosome
vector<int> generate_random_chromosome() {
    vector<int> chromosome(n);
    for(int i = 1; i <= n; i++) 
        chromosome[i-1] = i;
    random_shuffle(chromosome.begin(), chromosome.end());
    return chromosome;
}

// Generate greedy chromosome
vector<int> generate_greedy_chromosome() {
    vector<int> chromosome;
    vector<bool> used(n + 1, false);
    
    // Start with a random customer
    int curr = 1 + rand() % n;
    chromosome.push_back(curr);
    used[curr] = true;
    
    // Empirically decided radius (can be tuned)
    double radius = 1e8;
    
    while(chromosome.size() < n) {
        double min_dist = DBL_MAX;
        int next_customer = -1;
        
        // Find nearest unvisited customer within radius
        for(int i = 1; i <= n; i++) {
            if(!used[i] && d[curr][i] <= radius) {
                if(d[curr][i] < min_dist) {
                    min_dist = d[curr][i];
                    next_customer = i;
                }
            }
        }
        
        // If no customer found within radius, pick random unvisited
        if(next_customer == -1) {
            vector<int> remaining;
            for(int i = 1; i <= n; i++)
                if(!used[i]) remaining.push_back(i);
            next_customer = remaining[rand() % remaining.size()];
        }
        
        chromosome.push_back(next_customer);
        used[next_customer] = true;
        curr = next_customer;
    }
    
    return chromosome;
}

void generate_initial_population() {
    // Set random seed
    srand(time(NULL));
    
    // Calculate number of greedy individuals (10% of population)
    int greedy_count = max(1, (int)(0.1 * population_size));
    population.resize(population_size);
    
    // Generate 90% random individuals
    for(int i = 0; i < population_size; i++) {
        population[i].chromosome = generate_random_chromosome();
    }
    
    // Generate 10% greedy individuals
    /*for(int i = population_size - greedy_count; i < population_size; i++) {
        population[i].chromosome = generate_greedy_chromosome();
    }*/
    
    // Evaluate initial population
    for(auto& individual : population) {
        Solution sol = decode_chromosome(individual.chromosome);
        individual.fitness_vehicles = sol.num_vehicles;
        individual.fitness_distance = sol.total_distance;
        /*cout << sol.num_vehicles << " " << sol.total_distance << "\n";
        for (auto &i: sol.vehicles) {
            for (auto &j: i.route)
                cout << j << " ";
            cout << "\n";
        }
        cout << "\n\n\n";*/
    }
}

// Function to check if solution a dominates solution b
bool dominates(const Individual& a, const Individual& b) {
    // For both objectives (vehicles and distance), a should be <= b
    // And for at least one objective, a should be < b
    bool at_least_one_better = false;
    
    // Check vehicles
    if (a.fitness_vehicles > b.fitness_vehicles) return false;
    if (a.fitness_vehicles < b.fitness_vehicles) at_least_one_better = true;
    
    // Check distance
    if (a.fitness_distance > b.fitness_distance) return false;
    if (a.fitness_distance < b.fitness_distance) at_least_one_better = true;
    
    return at_least_one_better;
}

// Compute Pareto ranks for the population
void compute_pareto_ranks() {
    int population_size = population.size();
    vector<bool> ranked(population_size, false);
    int current_rank = 1;
    
    while (true) {
        vector<int> current_front;
        
        // Find all non-dominated solutions among the unranked ones
        for (int i = 0; i < population_size; i++) {
            if (ranked[i]) continue;
            
            bool is_dominated = false;
            for (int j = 0; j < population_size; j++) {
                if (i == j || ranked[j]) continue;
                
                if (dominates(population[j], population[i])) {
                    is_dominated = true;
                    break;
                }
            }
            
            if (!is_dominated) {
                current_front.push_back(i);
            }
        }
        
        // If no unranked non-dominated solutions found, we're done
        if (current_front.empty()) break;
        
        // Assign current rank to this front
        for (int idx : current_front) {
            population[idx].rank = current_rank;
            ranked[idx] = true;
        }
        
        current_rank++;
    }
}

// Tournament selection based on Pareto rank
int tournament_selection() {
    const int TOURNAMENT_SIZE = 4;
    const double TOURNAMENT_PROB = 0.8; // Probability to select the best
    
    vector<int> tournament;
    // Select TOURNAMENT_SIZE random individuals
    for (int i = 0; i < TOURNAMENT_SIZE; i++) {
        tournament.push_back(rand() % population.size());
    }
    
    // Find the best rank in tournament
    int best_idx = tournament[0];
    for (int i = 1; i < TOURNAMENT_SIZE; i++) {
        if (population[tournament[i]].rank < population[best_idx].rank) {
            best_idx = tournament[i];
        }
    }
    
    // Return best with probability TOURNAMENT_PROB, otherwise return random participant
    if ((double)rand() / RAND_MAX < TOURNAMENT_PROB) {
        return best_idx;
    } else {
        return tournament[rand() % TOURNAMENT_SIZE];
    }
}
// Helper function to find best insertion point for a customer
bool find_best_insertion(Vehicle& vehicle, int customer, int& best_pos, double& best_cost) {
    // If vehicle is empty, create new route
    if (vehicle.route.empty()) {
        vehicle.route = {0, customer, 0};
        vehicle.arrivals = {0};
        if (!vehicle.update_times(1)) return false;
        if (vehicle.update_load(1) > Q) return false;
        vehicle.update_distance(1);
        best_pos = 1;
        best_cost = vehicle.traveled_distance;
        return true;
    }

    best_cost = DBL_MAX;
    best_pos = -1;
    
    // Try inserting at each position between depot and depot
    for (int i = 1; i < vehicle.route.size(); i++) {
        Vehicle temp = vehicle;
        temp.route.insert(temp.route.begin() + i, customer);
        temp.arrivals.clear();
        temp.load = 0;
        temp.traveled_distance = 0;
        
        if (!temp.update_times(1)) continue;
        if (temp.update_load(1) > Q) continue;
        temp.update_distance(1);
        
        if (temp.traveled_distance < best_cost) {
            best_cost = temp.traveled_distance;
            best_pos = i;
        }
    }
    
    return best_pos != -1;
}

pair<Individual, Individual> crossover(const Individual& p1, const Individual& p2) {
    Solution s1 = decode_chromosome(p1.chromosome);
    Solution s2 = decode_chromosome(p2.chromosome);
    
    // Create both offspring starting as copies of parents
    Solution offspring1 = s1;
    Solution offspring2 = s2;
    
    // Select random route from each parent
    int route1_idx = rand() % s1.vehicles.size();
    int route2_idx = rand() % s2.vehicles.size();
    
    Vehicle& route1 = s1.vehicles[route1_idx];
    Vehicle& route2 = s2.vehicles[route2_idx];
    
    if (route1.route.size() <= 2 || route2.route.size() <= 2) {
        // If either route is invalid/empty, return parents
        return {p1, p2};
    }

    // Get customers from selected routes
    vector<int> customers_route1, customers_route2;
    for (int i = 1; i < route1.route.size() - 1; i++) {
        customers_route1.push_back(route1.route[i]);
    }
    for (int i = 1; i < route2.route.size() - 1; i++) {
        customers_route2.push_back(route2.route[i]);
    }
    
    // For offspring1: Remove customers from route2 and reinsert them
    for (int customer : customers_route2) {
        // Remove customer from any route in offspring1
        for (Vehicle& v : offspring1.vehicles) {
            auto it = find(v.route.begin(), v.route.end(), customer);
            if (it != v.route.end()) {
                v.route.erase(it);
                if (v.route.size() <= 2) {
                    v.route.clear();
                } else {
                    v.arrivals.clear();
                    v.load = 0;
                    v.traveled_distance = 0;
                    v.update_times(1);
                    v.update_load(1);
                    v.update_distance(1);
                }
                break;
            }
        }
    }

    // For offspring2: Remove customers from route1 and reinsert them
    for (int customer : customers_route1) {
        // Remove customer from any route in offspring2
        for (Vehicle& v : offspring2.vehicles) {
            auto it = find(v.route.begin(), v.route.end(), customer);
            if (it != v.route.end()) {
                v.route.erase(it);
                if (v.route.size() <= 2) {
                    v.route.clear();
                } else {
                    v.arrivals.clear();
                    v.load = 0;
                    v.traveled_distance = 0;
                    v.update_times(1);
                    v.update_load(1);
                    v.update_distance(1);
                }
                break;
            }
        }
    }

    // Clean both offspring
    auto clean_solution = [](Solution& sol) {
        vector<Vehicle> valid_vehicles;
        for (const Vehicle& v : sol.vehicles) {
            if (!v.route.empty()) {
                valid_vehicles.push_back(v);
            }
        }
        sol.vehicles = valid_vehicles;
    };

    clean_solution(offspring1);
    clean_solution(offspring2);

    // Reinsert customers in offspring1
    for (int customer : customers_route2) {
        double best_global_cost = DBL_MAX;
        int best_vehicle = -1;
        int best_position = -1;
        
        // Try insertion in each vehicle
        for (int v = 0; v < offspring1.vehicles.size(); v++) {
            int best_pos;
            double best_cost;
            Vehicle temp = offspring1.vehicles[v];
            
            if (find_best_insertion(temp, customer, best_pos, best_cost)) {
                if (best_cost < best_global_cost) {
                    best_global_cost = best_cost;
                    best_vehicle = v;
                    best_position = best_pos;
                }
            }
        }
        
        if (best_vehicle == -1) {
            Vehicle new_vehicle;
            new_vehicle.id = offspring1.vehicles.size();
            new_vehicle.route = {0, customer, 0};
            new_vehicle.arrivals = {0};
            if (new_vehicle.update_times(1) && new_vehicle.update_load(1) <= Q) {
                new_vehicle.update_distance(1);
                offspring1.vehicles.push_back(new_vehicle);
            }
        } else {
            offspring1.vehicles[best_vehicle].route.insert(
                offspring1.vehicles[best_vehicle].route.begin() + best_position, 
                customer
            );
            offspring1.vehicles[best_vehicle].arrivals.clear();
            offspring1.vehicles[best_vehicle].load = 0;
            offspring1.vehicles[best_vehicle].traveled_distance = 0;
            offspring1.vehicles[best_vehicle].update_times(1);
            offspring1.vehicles[best_vehicle].update_load(1);
            offspring1.vehicles[best_vehicle].update_distance(1);
        }
    }

    // Reinsert customers in offspring2 (same process)
    for (int customer : customers_route1) {
        double best_global_cost = DBL_MAX;
        int best_vehicle = -1;
        int best_position = -1;
        
        for (int v = 0; v < offspring2.vehicles.size(); v++) {
            int best_pos;
            double best_cost;
            Vehicle temp = offspring2.vehicles[v];
            
            if (find_best_insertion(temp, customer, best_pos, best_cost)) {
                if (best_cost < best_global_cost) {
                    best_global_cost = best_cost;
                    best_vehicle = v;
                    best_position = best_pos;
                }
            }
        }
        
        if (best_vehicle == -1) {
            Vehicle new_vehicle;
            new_vehicle.id = offspring2.vehicles.size();
            new_vehicle.route = {0, customer, 0};
            new_vehicle.arrivals = {0};
            if (new_vehicle.update_times(1) && new_vehicle.update_load(1) <= Q) {
                new_vehicle.update_distance(1);
                offspring2.vehicles.push_back(new_vehicle);
            }
        } else {
            offspring2.vehicles[best_vehicle].route.insert(
                offspring2.vehicles[best_vehicle].route.begin() + best_position, 
                customer
            );
            offspring2.vehicles[best_vehicle].arrivals.clear();
            offspring2.vehicles[best_vehicle].load = 0;
            offspring2.vehicles[best_vehicle].traveled_distance = 0;
            offspring2.vehicles[best_vehicle].update_times(1);
            offspring2.vehicles[best_vehicle].update_load(1);
            offspring2.vehicles[best_vehicle].update_distance(1);
        }
    }

    // Convert solutions back to chromosomes
    pair<Individual, Individual> result;
    
    // For offspring1
    result.first.chromosome.clear();
    for (const Vehicle& v : offspring1.vehicles) {
        for (int i = 1; i < v.route.size() - 1; i++) {
            result.first.chromosome.push_back(v.route[i]);
        }
    }

    
    // For offspring2
    result.second.chromosome.clear();
    for (const Vehicle& v : offspring2.vehicles) {
        for (int i = 1; i < v.route.size() - 1; i++) {
            result.second.chromosome.push_back(v.route[i]);
        }
    }
    
    // Evaluate both solutions
    Solution final_sol1 = decode_chromosome(result.first.chromosome);
    result.first.fitness_vehicles = final_sol1.num_vehicles;
    result.first.fitness_distance = final_sol1.total_distance;
    
    Solution final_sol2 = decode_chromosome(result.second.chromosome);
    result.second.fitness_vehicles = final_sol2.num_vehicles;
    result.second.fitness_distance = final_sol2.total_distance;
    
    return result;
}

Individual mutate(const Individual& ind) {
    // First decode the chromosome to get the routes
    Solution sol = decode_chromosome(ind.chromosome);
    
    // If no valid routes, return unchanged
    if (sol.vehicles.empty() || sol.vehicles[0].route.size() <= 2) {
        return ind;
    }

    // Select a random route that has at least 3 customers
    vector<int> valid_routes;
    for (int i = 0; i < sol.vehicles.size(); i++) {
        if (sol.vehicles[i].route.size() > 4) {  // depot + at least 3 customers + depot
            valid_routes.push_back(i);
        }
    }
    
    if (valid_routes.empty()) return ind;
    
    int route_idx = valid_routes[rand() % valid_routes.size()];
    Vehicle& route = sol.vehicles[route_idx];
    
    // Select segment length (2 or 3 customers)
    int seg_length = 2 + (rand() % 2);  // randomly choose 2 or 3
    
    // Select start position for reversal (must leave room for segment)
    int max_start = route.route.size() - seg_length - 1;  // -1 for depot
    if (max_start <= 1) return ind;  // not enough room for reversal
    
    int start_pos = 1 + (rand() % (max_start - 1));  // ensure we start after depot
    
    // Create temporary route with reversed segment
    Vehicle temp_route = route;
    reverse(temp_route.route.begin() + start_pos, 
           temp_route.route.begin() + start_pos + seg_length);
    
    // Check if new route is feasible
    temp_route.arrivals.clear();
    temp_route.load = 0;
    temp_route.traveled_distance = 0;
    
    if (!temp_route.update_times(1)) return ind;
    if (temp_route.update_load(1) > Q) return ind;
    temp_route.update_distance(1);
    
    // If feasible and improves distance, apply mutation
    if (temp_route.traveled_distance < route.traveled_distance) {
        sol.vehicles[route_idx] = temp_route;
        
        // Create new chromosome from modified solution
        Individual result;
        result.chromosome.clear();
        for (const Vehicle& v : sol.vehicles) {
            for (int i = 1; i < v.route.size() - 1; i++) {
                result.chromosome.push_back(v.route[i]);
            }
        }
        
        // Evaluate new solution
        Solution final_sol = decode_chromosome(result.chromosome);
        result.fitness_vehicles = final_sol.num_vehicles;
        result.fitness_distance = final_sol.total_distance;
        
        return result;
    }
    
    return ind;
}
// Function to create next generation
double CROSSOVER_RATE = 0.8;  // 80% chance of crossover
double MUTATION_RATE = 0.1;   // 10% chance of mutation

vector<Individual> create_next_generation() {
   vector<Individual> next_generation;
   
   // First, compute Pareto ranks for current population
   compute_pareto_ranks();
   
   // Keep track of best solutions (rank 1)
   vector<Individual> elite;
   for (const Individual& ind : population) {
       if (ind.rank == 1) {
           elite.push_back(ind);
       }
   }
   
   // Add elite solutions to next generation
   for (const Individual& ind : elite) {
       next_generation.push_back(ind);
   }
   
   // Fill rest of population through selection, crossover and mutation
   while (next_generation.size() < population_size) {
       // Select parents
       int parent1_idx = tournament_selection();
       int parent2_idx = tournament_selection();
       
       pair<Individual, Individual> offspring;
       
       // Apply crossover with probability CROSSOVER_RATE
       if ((double)rand() / RAND_MAX < CROSSOVER_RATE) {
           offspring = crossover(population[parent1_idx], population[parent2_idx]);
       } else {
           offspring = {population[parent1_idx], population[parent2_idx]};
       }
       
       // Apply mutation with probability MUTATION_RATE to each offspring
       if ((double)rand() / RAND_MAX < MUTATION_RATE) {
           offspring.first = mutate(offspring.first);
       }
       if ((double)rand() / RAND_MAX < MUTATION_RATE) {
           offspring.second = mutate(offspring.second);
       }
       
       // Add both offspring to next generation if there's room
       next_generation.push_back(offspring.first);
       if (next_generation.size() < population_size) {
           next_generation.push_back(offspring.second);
       }
   }
   
   // Ensure we don't exceed population size
   while (next_generation.size() > population_size) {
       next_generation.pop_back();
   }
   
   return next_generation;
}

void print_generation_stats(int generation) {
   // Find rank 1 solutions
   compute_pareto_ranks();
   vector<Individual> pareto_front;
   for (const Individual& ind : population) {
       if (ind.rank == 1) {
           pareto_front.push_back(ind);
       }
   }
   
   // Sort by number of vehicles
   sort(pareto_front.begin(), pareto_front.end(), 
        [](const Individual& a, const Individual& b) {
            return a.fitness_vehicles < b.fitness_vehicles || 
                   (a.fitness_vehicles == b.fitness_vehicles && 
                    a.fitness_distance < b.fitness_distance);
        });
   
   cout << "\nGeneration " << generation << ":\n";
   /*cout << "Pareto front size: " << pareto_front.size() << "\n";
   cout << "Non-dominated solutions:\n";
   for (const Individual& ind : pareto_front) {
       cout << "Vehicles: " << ind.fitness_vehicles 
            << ", Distance: " << fixed << setprecision(2) 
            << ind.fitness_distance << "\n";
   }*/
   
   // Find best solution for each objective
   auto min_vehicles = min_element(pareto_front.begin(), pareto_front.end(),
       [](const Individual& a, const Individual& b) {
           return a.fitness_vehicles < b.fitness_vehicles;
       });
   
   auto min_distance = min_element(pareto_front.begin(), pareto_front.end(),
       [](const Individual& a, const Individual& b) {
           return a.fitness_distance < b.fitness_distance;
       });
   
   cout << "\nBest solutions:\n";
   cout << "Min vehicles: " << min_vehicles->fitness_vehicles 
        << " (distance: " << fixed << setprecision(2) 
        << min_vehicles->fitness_distance << ")\n";
   cout << "Min distance: " << min_distance->fitness_vehicles 
        << " (distance: " << min_distance->fitness_distance << ")\n";
   cout << "----------------------------------------\n";
}


// Main evolutionary loop
void evolutionary_algorithm(int max_generations, int population_size, double crossover_rate, double mutation_rate) {
    // Set the global constants based on the function parameters
    ::population_size = population_size;
    ::CROSSOVER_RATE = crossover_rate;
    ::MUTATION_RATE = mutation_rate;

    generate_initial_population();
    
    // Track best solution
    Individual best_individual = population[0];
    for(const Individual& ind : population) {
        if(ind.fitness_vehicles < best_individual.fitness_vehicles || 
           (ind.fitness_vehicles == best_individual.fitness_vehicles && 
            ind.fitness_distance < best_individual.fitness_distance)) {
            best_individual = ind;
        }
    }
    
    //cout << "Initial population:\n";
    print_generation_stats(0);
    
    // Main loop
    auto start_time = chrono::high_resolution_clock::now();
    
    for (int gen = 1; gen <= max_generations; gen++) {
        vector<Individual> next_gen = create_next_generation();
        population = next_gen;
        
        // Update best solution
        for(const Individual& ind : population) {
            if(ind.fitness_vehicles < best_individual.fitness_vehicles || 
               (ind.fitness_vehicles == best_individual.fitness_vehicles && 
                ind.fitness_distance < best_individual.fitness_distance)) {
                best_individual = ind;
            }
        }
        
        //Print statistics every generation
        print_generation_stats(gen);
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    // Get final solution from best individual
    Solution best_solution = decode_chromosome(best_individual.chromosome);
    best_solution.computation_time = duration.count();

    //cout << "\nBest solution found:\n";
    out_solution(best_solution);
}
int main() {
    string instance = "18";
    string base_directory = get_current_directory();
    string instance_path = base_directory + "\\VRPTW Instances\\VRPTW" + instance + ".txt";
   
    read_instance(instance_path);
    compute_distances();
    evolutionary_algorithm(200, 1000, 0.5, 0.5);

    return 0;
}

/*int main(int argc, char* argv[]) {
    string instance = argv[1];
    int time_limit = stoi(argv[2]);
    
    string base_directory = get_current_directory();
    string instance_path = base_directory + "\\VRPTW Instances\\VRPTW" + instance + ".txt";
    read_instance(instance_path);
    compute_distances();
    evolutionary_algorithm(250, 1000, 0.9, 0.1);
    
    return 0;
}*/
