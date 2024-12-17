#include <bits/stdc++.h>
#include <omp.h>
#include <windows.h>
using namespace std;

#define sz(x) (int)x.size()
#define all(x) x.begin(), x.end()
#define rall(x) x.rbegin(), x.rend()
#define pb push_back

double d[101][101] = {0.0};
bool is_there_edge[101][101] = {false};
int n, Q;

string getCurrentDirectory() {
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

    bool update_times(int beginning) {
        for (int i = beginning; i < sz(arrivals); i++) {
            const Node& prev_node = nodes[route[i - 1]];
            const Node& curr_node = nodes[route[i]];

            double current_time = arrivals[i - 1] + prev_node.s + d[route[i - 1]][route[i]];
            if (current_time > curr_node.l) return false;
            arrivals[i] = max(current_time, curr_node.e);
        }
        return true;
    }

    double update_load(int beginning) {
        double acum_from = 0;
        for (int i = beginning; i < sz(route); i++)
            acum_from += nodes[route[i]].q;
        load += acum_from;
        return acum_from;
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

void write_solution(Solution& sol, const string& file_path) {
    FILE* file = freopen(file_path.c_str(), "w", stdout);

    cout << sol.num_vehicles << " " << sol.total_distance << " " << sol.computation_time;

    for (int i = 0; i < sol.num_vehicles; i++) {
        cout << "\n" << sz(sol.vehicles[i].route) - 2 << " ";
        for (auto& j : sol.vehicles[i].route)
            cout << j << " ";
        for (auto& j : sol.vehicles[i].arrivals)
            cout << j << " ";
        cout << sol.vehicles[i].load;
    }

    fclose(file);
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

void dfs(vector<vector<pair<double, int>>>& adj, int i, double current_time, double current_dist, int current_load, vector<bool>& visited, Vehicle& vehicle) {
    current_time = max(current_time, nodes[i].e);
    vehicle.route.pb(i);
    vehicle.arrivals.pb(current_time);
    current_time += nodes[i].s;

    if (i==0 && visited[0]==true) {
        vehicle.traveled_distance = current_dist;
        vehicle.load = current_load;
        return;
    }

    visited[i] = true;
    bool to_zero = true;
    
    for (auto &[ _, j]: adj[i]) {
        if (j==0 || visited[j]) continue;
        if ( (current_time + d[i][j] <= nodes[j].l && current_load + nodes[j].q <= Q) && (max(current_time + d[i][j], nodes[j].e) + nodes[j].s + d[0][j])<= nodes[0].l)  {
            to_zero = false;
            dfs(adj, j, current_time + d[i][j], current_dist + d[i][j], current_load + nodes[j].q, visited, vehicle);
            return;
        }
    }

    if (to_zero) dfs(adj, 0, current_time + d[i][0], current_dist + d[i][0], current_load, visited, vehicle);
}

Solution generate_initial_solution(unsigned int seed) {
    random_device rd;
    default_random_engine generator(seed);
    vector<vector<pair<double, int>>> adj;
    adj.resize(n + 1);
    mt19937 gen(rd());

    uniform_real_distribution<> dist(-10, 10);

    double alpha = 0.5;
    double Beta = 0.5;


    // Define generate_noise as a lambda
    auto generate_noise = [&generator](double noiseRange) -> double {
        if (noiseRange == 0) return 0;
        if (noiseRange < 0) {
            uniform_real_distribution<double> distribution(noiseRange, 0);
            return distribution(generator);
        }
        uniform_real_distribution<double> distribution(0, noiseRange);
        return distribution(generator);
    };

    // Define make_graph as a lambda
    auto make_graph = [&](vector<vector<pair<double, int>>>& adj, double Beta, double alpha) -> vector<vector<pair<double, int>>> {
        double* max_ptr = max_element(d[0], d[0] + 101);
        double dmax = *max_ptr;

        for (int i = 0; i <= n; i++) {
            double earliest = max(d[0][i], nodes[i].e) + nodes[i].s;
            for (int j = 0; j <= n; j++) {
                if (i == j) continue;
                if (j == 0 || (earliest + d[i][j] <= nodes[j].l && earliest + d[i][j] + nodes[j].s + d[j][0] <= nodes[0].l && nodes[i].q + nodes[j].q <= Q)) {
                    is_there_edge[i][j] = true;
                    if (i == 0) {
                        adj[i].pb(make_pair(Beta * d[i][j] - (1 - Beta) * nodes[j].l, j));
                    } else {
                        double miu = (d[0][i] / dmax * d[0][j] / dmax) * (d[i][j] - d[j][0]);
                        double noise = generate_noise(miu);
                        adj[i].pb(make_pair(alpha * d[i][j] + (1 - alpha) * nodes[j].e + noise, j));
                    }
                }
            }
            sort(all(adj[i]));
        }

        for (int j = 0; j <= n; j++) {
            adj[0][j].first -= 1e9 * (sz(adj[adj[0][j].second]) == 1);
        }

        sort(rall(adj[0]));
        return adj;
    };
    // Call make_graph
    adj = make_graph(adj, Beta, alpha);

    Solution sol;
    sol.num_vehicles = 0;
    sol.total_distance = 0.0;
    vector<bool> visited(n + 1, false);

    // Main loop
    while (count(all(visited), false) > 1) {
        Vehicle vehicle;
        vehicle.id = sol.num_vehicles;
        dfs(adj, 0, 0, 0, 0, visited, vehicle);
        visited[0] = false;
        sol.total_distance += vehicle.traveled_distance;
        sol.vehicles.pb(vehicle);
        sol.num_vehicles++;
    }

    return sol;
}

double compute_cost_1(const Solution& sol) {
    double total_distance = sol.total_distance;
    int num_vehicles = sol.num_vehicles;
    
    // Find e_min (number of customers in the smallest route)
    int e_min = INT_MAX;
    for (const Vehicle& v : sol.vehicles) {
        int customers_in_route = v.route.size() - 2;  // Subtract 2 for depot start/end
        e_min = min(e_min, customers_in_route);
    }
    // cost(s) = d + sigma * (cn + e_min)
    double total_cost = total_distance + 1e8 *num_vehicles +  1e5 * e_min;

    return total_cost;
}

double compute_cost_2(const Solution& sol) {
    double total_distance = sol.total_distance;
    int num_vehicles = sol.num_vehicles;
    double total_cost = total_distance + 1e8 *num_vehicles;
    return total_cost;
}

Solution shift_op(Solution &sol, Vehicle &v1, Vehicle &v2, int idx1, int idx2) {
    // Copy the vehicles in the new solution
    Vehicle new_v1 = sol.vehicles[v1.id];
    Vehicle new_v2 = sol.vehicles[v2.id];

    int current1 = v1.route[idx1];
    int next1 = v1.route[idx1+1];
    int prev2 = v2.route[idx2-1];
    int current2 = v2.route[idx2];
    int next2 = v2.route[idx2+1];



    // The node to be moved
    int node_to_move = new_v2.route[idx2];

    // Remove the node from v2's route
    new_v2.route.erase(new_v2.route.begin() + idx2);
    new_v2.arrivals.erase(new_v2.arrivals.begin() + idx2);

    // Update the load of v2
    new_v2.load -= nodes[node_to_move].q;

    // Insert the node into v1's route at position idx1 + 1
    new_v1.route.insert(new_v1.route.begin() + idx1 + 1, node_to_move);
    new_v1.arrivals.insert(new_v1.arrivals.begin() + idx1 + 1, 0.0);

    // Update the load of v1
    new_v1.load += nodes[node_to_move].q;

    // Recalculate the times and check time windows for v1
    if (new_v1.load > Q || !new_v1.update_times(1) || !new_v2.update_times(1))
        return sol;

    Solution new_sol = sol;


    new_v1.traveled_distance = 0;
    new_v2.traveled_distance = 0;
    new_v1.update_distance(1);
    new_v2.update_distance(1);

    double delta = new_v1.traveled_distance + new_v2.traveled_distance - v1.traveled_distance - v2.traveled_distance;


    new_sol.total_distance += delta;
    new_sol.vehicles[v1.id] = new_v1;
    new_sol.vehicles[v2.id] = new_v2;

    if (sz(new_v2.route) == 2 && new_v2.route[0] == 0 && new_v2.route[1] == 0) {
        // Remove the vehicle from the solution
        new_sol.vehicles.erase(new_sol.vehicles.begin() + v2.id);
        new_sol.num_vehicles--; // Update the number of vehicles

        for (int i = v2.id; i < sz(new_sol.vehicles); i++)
            new_sol.vehicles[i].id = i;  // Update the ID of each vehicle
    }

    // Return the new solution if all constraints are satisfied
    return new_sol;
}

Solution perturb_solution_1(Solution& current_solution, mt19937& gen) {
    int num_vehicles = current_solution.num_vehicles;
    uniform_int_distribution<> vehicle_dist(0, num_vehicles - 1);

    int v1_id = vehicle_dist(gen);
    int v2_id = vehicle_dist(gen);
    do {
        v2_id = vehicle_dist(gen);
    } while (v1_id == v2_id);

    Vehicle& v1 = current_solution.vehicles[v1_id];
    Vehicle& v2 = current_solution.vehicles[v2_id];

    if (sz(v1.route) > 2 && sz(v2.route) > 2) {
        uniform_int_distribution<> route_dist1(1, sz(v1.route) - 2);
        uniform_int_distribution<> route_dist2(1, sz(v2.route) - 2);
        int idx1 = route_dist1(gen);
        int idx2 = route_dist2(gen);
        Solution new_solution = shift_op(current_solution, v1, v2, idx1, idx2);
        return new_solution;
    }

    return current_solution;
}

Solution rearrange_op(Solution &sol, Vehicle &v, int start_idx, int end_idx, int insert_position) {
    Vehicle new_v = v;

    vector<int> segment(new_v.route.begin() + start_idx, new_v.route.begin() + end_idx + 1);

    new_v.route.erase(new_v.route.begin() + start_idx, new_v.route.begin() + end_idx + 1);

    if (insert_position > start_idx) insert_position -= (end_idx - start_idx + 1);
    
    new_v.route.insert(new_v.route.begin() + insert_position, segment.begin(), segment.end());

    if (!new_v.update_times(1)) return sol;
    new_v.traveled_distance = 0;
    new_v.update_distance(1);

    Solution new_sol = sol;
    new_sol.vehicles[v.id] = new_v;

    new_sol.total_distance += new_v.traveled_distance - v.traveled_distance;
    
    return new_sol;
}

Solution perturb_solution_2(Solution& current_solution, mt19937& gen) {
    int num_vehicles = current_solution.num_vehicles;

    // Distribution to select a vehicle at random
    uniform_int_distribution<> vehicle_dist(0, num_vehicles - 1);

    // Select a vehicle randomly
    int v_id = vehicle_dist(gen);
    Vehicle& v = current_solution.vehicles[v_id];

    // Ensure the vehicle has enough nodes (at least 3 nodes for a valid segment)
    if (sz(v.route) > 2) {
        // Distribution to select start and end indices for a segment
        uniform_int_distribution<> segment_dist(1, sz(v.route) - 2);  // Avoid depot nodes at 0 and sz-1

        int start_idx = segment_dist(gen);  // Random start index for the segment
        int end_idx = segment_dist(gen);    // Random end index for the segment

        // Ensure that start_idx is less than or equal to end_idx
        if (start_idx > end_idx) swap(start_idx, end_idx);

        // Distribution to select the insertion position
        uniform_int_distribution<> insert_dist(1, sz(v.route) - 2);  // Avoid depot

        int insert_position = insert_dist(gen);

        // Adjust insert_position if it intersects with the segment [start_idx, end_idx]
        if (insert_position >= start_idx && insert_position <= end_idx) {
            // If insertion is within the segment, move the insert_position to after the segment
            insert_position = end_idx + 1;
            // If this makes it exceed the route size, wrap it around to a valid position
            if (insert_position >= sz(v.route)) insert_position = start_idx - 1;
        }

        // Perform the rearrange operation using the rearrange_op function
        Solution new_solution = rearrange_op(current_solution, v, start_idx, end_idx, insert_position);

        // Return the new solution
        return new_solution;
    }

    // If the vehicle doesn't have enough nodes for a valid rearrangement, return the original solution
    return current_solution;
}

Solution exchange_op(Solution &sol, Vehicle &v1, Vehicle &v2, int idx1, int idx2) {

    Vehicle new_v1 = sol.vehicles[v1.id];
    Vehicle new_v2 = sol.vehicles[v2.id];

    int current1 = v1.route[idx1];
    int next1 = v1.route[idx1+1];

    int current2 = v2.route[idx2];
    int next2 = v2.route[idx2+1];

    vector<int> v1_segment(new_v1.route.begin() + idx1 + 1, new_v1.route.end());
    vector<int> v2_segment(new_v2.route.begin() + idx2 + 1, new_v2.route.end());
    vector<double> a1_segment(new_v1.arrivals.begin() + idx1 + 1, new_v1.arrivals.end());
    vector<double> a2_segment(new_v2.arrivals.begin() + idx2 + 1, new_v2.arrivals.end());

    new_v1.route.erase(new_v1.route.begin() + idx1 + 1, new_v1.route.end());
    new_v2.route.erase(new_v2.route.begin() + idx2 + 1, new_v2.route.end());
    new_v1.arrivals.erase(new_v1.arrivals.begin() + idx1 + 1, new_v1.arrivals.end());
    new_v2.arrivals.erase(new_v2.arrivals.begin() + idx2 + 1, new_v2.arrivals.end());


    new_v1.route.insert(new_v1.route.begin() + idx1+1, v2_segment.begin(), v2_segment.end());
    new_v2.route.insert(new_v2.route.begin() + idx2+1, v1_segment.begin(), v1_segment.end());
    new_v1.arrivals.insert(new_v1.arrivals.begin() + idx1+1, a2_segment.begin(), a2_segment.end());
    new_v2.arrivals.insert(new_v2.arrivals.begin() + idx2+1, a1_segment.begin(), a1_segment.end());

    if (!new_v1.update_times(1) || !new_v2.update_times(1))
        return sol;

    new_v2.load = new_v2.update_load(1);
    new_v1.load = new_v1.update_load(1);
    if (new_v2.load > Q || new_v1.load > Q)
        return sol;


    /*new_v2.traveled_distance -= new_v1.update_distance(idx1+1);
    new_v1.traveled_distance -= new_v2.update_distance(idx2+1);*/
    new_v1.traveled_distance = 0;
    new_v2.traveled_distance = 0;
    new_v1.update_distance(1);
    new_v2.update_distance(1);

    Solution new_sol = sol;

    new_sol.total_distance += new_v1.traveled_distance + new_v2.traveled_distance - v1.traveled_distance - v2.traveled_distance;

    new_sol.vehicles[v1.id] = new_v1;
    new_sol.vehicles[v2.id] = new_v2;

    if (sz(new_v2.route) == 2 && new_v2.route[0] == 0 && new_v2.route[1] == 0) {
        new_sol.vehicles.erase(new_sol.vehicles.begin() + v2.id);
        new_sol.num_vehicles--;

        for (int i = v2.id; i < sz(new_sol.vehicles); i++) {
            if (new_sol.vehicles[i].id==new_v1.id) new_v1.id = i;
            new_sol.vehicles[i].id = i;
        }
    }

    if (sz(new_v1.route) == 2 && new_v1.route[0] == 0 && new_v1.route[1] == 0) {
        new_sol.vehicles.erase(new_sol.vehicles.begin() + new_v1.id);
        new_sol.num_vehicles--;

        for (int i = new_v1.id; i < sz(new_sol.vehicles); i++)
            new_sol.vehicles[i].id = i; 
    }

    return new_sol;
}

Solution perturb_solution_3(Solution& current_solution, mt19937& gen) {
    int num_vehicles = current_solution.num_vehicles;

    // Distribution to select two vehicles at random
    uniform_int_distribution<> vehicle_dist(0, num_vehicles - 1);

    // Select two distinct vehicles randomly
    int v1_id = vehicle_dist(gen);
    int v2_id = vehicle_dist(gen);
    do {
        v2_id = vehicle_dist(gen);
    } while (v1_id == v2_id);  // Ensure v2 is different from v1

    Vehicle& v1 = current_solution.vehicles[v1_id];
    Vehicle& v2 = current_solution.vehicles[v2_id];

    // Ensure both vehicles have enough nodes (at least 2 non-depot nodes)
    if (sz(v1.route) > 2 && sz(v2.route) > 2) {
        // Distribution to select valid indices for the exchange (avoid depot nodes at 0 and sz-1)
        uniform_int_distribution<> index_dist1(1, sz(v1.route) - 2);
        uniform_int_distribution<> index_dist2(1, sz(v2.route) - 2);

        int idx1 = index_dist1(gen);  // Random index for vehicle v1
        int idx2 = index_dist2(gen);  // Random index for vehicle v2

        // Ensure that the segments do not overlap or cause issues (optional: add constraints)
        // Here we are not exchanging if v1 and v2 indices are too close, but this can be customized further

        // Perform the exchange operation using the exchange_op function
        Solution new_solution = exchange_op(current_solution, v1, v2, idx1, idx2);

        // Return the new solution
        return new_solution;
    }

    // If the vehicles don't have enough nodes for a valid exchange, return the original solution
    return current_solution;
}

Solution SWAP(Solution& sol, Vehicle& v1, Vehicle& v2, int idx1, int idx2) {
    Vehicle new_v1 = v1;
    Vehicle new_v2 = v2;

    // Perform the swap
    int node1 = new_v1.route[idx1];
    int node2 = new_v2.route[idx2];

    swap(new_v1.route[idx1], new_v2.route[idx2]);

    new_v1.load += nodes[node2].q - nodes[node1].q;
    new_v2.load += nodes[node1].q - nodes[node2].q;;

    // Check if the swap maintains the time window constraints and the capacity constraints
    if (!new_v1.update_times(1) || !new_v2.update_times(1) || new_v1.load > Q || new_v2.load > Q) {
        return sol;
    }

    // Recalculate the distances after the swap
    new_v1.traveled_distance = 0;
    new_v2.traveled_distance = 0;
    new_v1.update_distance(1);
    new_v2.update_distance(1);

    // Create a new solution with updated vehicles
    Solution new_sol = sol;
    new_sol.vehicles[v1.id] = new_v1;
    new_sol.vehicles[v2.id] = new_v2;

    // Update the total distance of the new solution
    new_sol.total_distance += new_v1.traveled_distance + new_v2.traveled_distance;
    new_sol.total_distance -= (v1.traveled_distance + v2.traveled_distance);
    return new_sol;
}

Solution perturb_solution_4(Solution& current_solution, mt19937& gen) {
    int num_vehicles = current_solution.num_vehicles;

    // Distribution to select two distinct vehicles at random
    uniform_int_distribution<> vehicle_dist(0, num_vehicles - 1);

    int v1_id = vehicle_dist(gen);
    int v2_id;
    do {
        v2_id = vehicle_dist(gen);
    } while (v1_id == v2_id);  // Ensure v2 is different from v1

    Vehicle& v1 = current_solution.vehicles[v1_id];
    Vehicle& v2 = current_solution.vehicles[v2_id];

    // Select random valid indices for swapping nodes
    uniform_int_distribution<> index_dist1(1, sz(v1.route) - 2);
    uniform_int_distribution<> index_dist2(1, sz(v2.route) - 2);

    int idx1 = index_dist1(gen);  // Random index for vehicle v1
    int idx2 = index_dist2(gen);  // Random index for vehicle v2

    // Perform the swap operation between the two vehicles
    Solution new_solution = SWAP(current_solution, v1, v2, idx1, idx2);

    // Return the new solution after the swap operation
    return new_solution;
}

Solution two_opt(Solution& sol, Vehicle& v) {
    Vehicle new_v = v;
    bool improved = true;

    /*for (auto &c: v.route)
        cout << c << " ";
    cout << "\n";*/

    while (improved) {
        improved = false;

        for (int i = 1; i < sz(new_v.route) - 2; ++i) {
            for (int j = i + 1; j < sz(new_v.route) - 1; ++j) {
                reverse(new_v.route.begin() + i, new_v.route.begin() + j + 1);

                if (!new_v.update_times(1)) {
                    reverse(new_v.route.begin() + i, new_v.route.begin() + j + 1);
                    new_v.update_times(1);
                    continue;
                }

                new_v.traveled_distance = 0;
                new_v.update_distance(1);

                if (new_v.traveled_distance < v.traveled_distance) {
                    improved = true;
                    v.traveled_distance = new_v.traveled_distance;
                } else {
                    reverse(new_v.route.begin() + i, new_v.route.begin() + j + 1);
                    new_v.traveled_distance = 0;
                    new_v.update_distance(1);
                    new_v.update_times(1);
                }
            }
        }
    }
    /*for (auto &c: new_v.route)
        cout << c << " ";
    cout << "\nchao\n";*/

    // Return the updated solution
    Solution new_sol = sol;
    new_sol.vehicles[v.id] = new_v;
    new_sol.total_distance += new_v.traveled_distance - v.traveled_distance;
    return new_sol;
}

Solution perturb_solution_5(Solution& current_solution, mt19937& gen) {
    int num_vehicles = current_solution.num_vehicles;
    uniform_int_distribution<> vehicle_dist(0, num_vehicles - 1);
    int v1_id = vehicle_dist(gen);
    Vehicle& v1 = current_solution.vehicles[v1_id];
    Solution new_solution = two_opt(current_solution, v1);
    return new_solution;
}

Solution parallel_simulated_annealing(int num_threads, int K, double gamma, double BETA, int tau, int cooperation_interval, int zeta, int time_limit) {
    auto start_time = chrono::system_clock::now();
    vector<Solution> local_best(num_threads);     // Best solutions for each thread
    vector<double> temperatures(num_threads);     // Temperatures for each thread
    int equilibrium_counter = 0;                       // Counter to track equilibrium state
    bool go = true;                                    // Termination flag

    // Initialize local solutions and temperatures for each thread
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        random_device rd;
        unsigned int seed = rd();
        default_random_engine generator(seed);

        local_best[thread_id] = generate_initial_solution(seed);
        double initial_cost = compute_cost_1(local_best[thread_id]);
        temperatures[thread_id] = gamma * initial_cost;  // Set initial temperature based on cost

        #pragma omp critical
        cout << "Initial solution for thread " << thread_id << " with num_vehicles: " << local_best[thread_id].num_vehicles << " and total_distance: " << local_best[thread_id].total_distance << "\n";
    }

    Solution global_best = *min_element(local_best.begin(), local_best.end(), [](const Solution& a, const Solution& b) {
                            return compute_cost_2(a) < compute_cost_2(b);});

    // Start the simulated annealing process
    while (go) {
        #pragma omp parallel num_threads(num_threads) shared(global_best, local_best, equilibrium_counter, go, temperatures)
        {
            int thread_id = omp_get_thread_num();
            unsigned seed = chrono::system_clock::now().time_since_epoch().count() + thread_id;
            mt19937 gen(seed);

            for (int i = 0; i < K; ++i) {
                auto current_time = chrono::system_clock::now();
                chrono::duration<double> elapsed_seconds = current_time - start_time;

                if (elapsed_seconds.count()+1 >= time_limit) {
                    go = false;
                    break;
                }

                Solution new_solution = perturb_solution_4(local_best[thread_id], gen);
                new_solution = perturb_solution_1(new_solution, gen);
                new_solution = perturb_solution_2(new_solution, gen);
                new_solution = perturb_solution_3(new_solution, gen);
                new_solution = perturb_solution_5(new_solution, gen);

                double new_cost = compute_cost_1(new_solution);
                double current_cost = compute_cost_1(local_best[thread_id]);
                double delta = current_cost - new_cost;
                uniform_real_distribution<> random_prob(0.0, 1.0);
               
                if (new_cost != current_cost && (new_cost < current_cost || random_prob(gen) < (temperatures[thread_id] + delta) / temperatures[thread_id])) {
                    #pragma omp critical
                    {
                        local_best[thread_id] = new_solution;
                        cout << "Thread " << thread_id << " updated its solution with new cost: " << local_best[thread_id].num_vehicles << " " << local_best[thread_id].total_distance << "\n";
                    }
                }

                #pragma omp barrier

                // Cooperative phase: Sequentially pass solutions from left to right
                if (i % cooperation_interval == 0) {
                    for (int t = 1; t < num_threads; ++t) {
                        if (thread_id == t) {
                            Solution left_neighbor_solution = local_best[thread_id - 1];
                            if (compute_cost_1(left_neighbor_solution) < compute_cost_1(local_best[thread_id])) {
                                #pragma omp critical
                                {
                                    local_best[thread_id] = left_neighbor_solution;  // Accept the better solution from the left neighbor
                                    cout << "Thread " << thread_id << " received better solution from left neighbor (Thread " << thread_id - 1 << ")" << " " << local_best[thread_id].num_vehicles << " " << local_best[thread_id].total_distance << "\n";
                                }
                            }
                        }
                        #pragma omp barrier  // Ensure that each thread has updated before the next cooperation phase
                    }
                }
                
                if (equilibrium_counter >= zeta) {
                    double initial_cost = compute_cost_1(local_best[thread_id]);
                    temperatures[thread_id] = gamma * initial_cost;
                } else {
                    temperatures[thread_id] *= BETA;
                }

                #pragma omp barrier
            }

            #pragma omp single
            {
                Solution best_local_solution = *min_element(local_best.begin(), local_best.end(), [](const Solution& a, const Solution& b) {
                    return compute_cost_2(a) < compute_cost_2(b);
                });
                if (compute_cost_2(best_local_solution) < compute_cost_2(global_best)) {
                    global_best = best_local_solution;  // Update the global best solution
                    cout << "Global best updated with new cost: " << global_best.num_vehicles << " " << global_best.total_distance << "\n";
                    equilibrium_counter = 0;  // Reset the equilibrium counter when improvement is made
                } else {
                    equilibrium_counter++;  // Increment the equilibrium counter when no improvement
                    cout << "equilibrium_counter: " << equilibrium_counter << "\n";
                }

                if (equilibrium_counter >= tau) {
                    go = false;
                }
            }
        }
    }
    global_best.calculate_total_distance();
    return global_best;
}

/*int main(int argc, char* argv[]) {
    string instance = argv[1];
    int time_limit = stoi(argv[2]);
    string baseDirectory = getCurrentDirectory();
    string filePath1 = baseDirectory + "\\VRPTW Instances\\VRPTW" + instance + ".txt";
   
    read_instance(filePath1);
    compute_distances();
    random_device rd;
    unsigned int seed = rd();
    default_random_engine generator(seed);

    auto start = chrono::high_resolution_clock::now();

    int p = 8;
    int K = 100000;
    double gamma = 1;
    double BETA = 0.95; 
    int tau = 10;
    int omega = 30000;
    int zeta = 5;
    Solution best_solution = parallel_simulated_annealing(p, K, gamma, BETA, tau, omega, zeta, time_limit);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    best_solution.computation_time = round(duration.count());

    out_solution(best_solution);

    return 0;
}*/

int main() {
    string instance = "17";
    int time_limit = 200;
    string baseDirectory = getCurrentDirectory();
    string filePath1 = baseDirectory + "\\VRPTW Instances\\VRPTW" + instance + ".txt";
   
    read_instance(filePath1);
    compute_distances();
    random_device rd;
    unsigned int seed = rd();
    default_random_engine generator(seed);

    auto start = chrono::high_resolution_clock::now();

    int p = 8;
    int K = 1000;
    double gamma = 1;
    double BETA = 0.92; 
    int tau = 10;
    int omega = 100;
    int zeta = 5;
    Solution best_solution = parallel_simulated_annealing(p, K, gamma, BETA, tau, omega, zeta, time_limit);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    best_solution.computation_time = round(duration.count());

    out_solution(best_solution);

    return 0;
}
