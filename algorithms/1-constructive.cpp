#include <bits/stdc++.h>
using namespace std;

typedef long long       ll;
typedef long double     ld;

#define fi              first
#define se              second
#define sz(x)           (int) x.size()
#define all(x)          x.begin(), x.end()
#define rall(x)         x.rbegin(), x.rend()
#define pb              push_back

const double  INF = 1e8;

double alpha, Beta;
double dist_total = 0;
int n, Q, m=0, total_visited=0;
vector<vector<pair<double, int>>> adj;
vector<pair<int,int>> coord, timew;
vector<int> q, s;
vector<vector<double>> d;
vector<bool> visited;
vector<double> dmax;
vector<vector<double>> routes;
vector<vector<double>> times;
vector<double> capacities;
bool fin=false;
string instance;

double distance(pair<int,int> a, pair<int,int> b) {
    double dist = sqrt( (a.fi-b.fi)*(a.fi-b.fi) + (a.se-b.se)*(a.se-b.se) );
    return round(dist*1000)/1000;
}

void dfs(double current_time, double current_dist, int capacity, int current_node) {
    if (visited[current_node] || fin) return;
    routes[m].pb(current_node);
    times[m].pb(current_time);

    visited[current_node] = true;

    if (current_node == 0) {
        if (current_time == 0.0) {
            visited[current_node] = false;
        } else {
            dist_total += current_dist;
            capacities.pb(Q-capacity);
            fin = true;
            return;
        }
    } else {
        total_visited++;
    }

    current_time = max(current_time, (double)timew[current_node].fi) + s[current_node];

    bool to_zero = false;

    for (auto &[ _, neighbor]: adj[current_node]) {
        if (neighbor==0 || visited[neighbor]) continue;

        double dist = d[current_node][neighbor];

        if ( (current_time + dist <= timew[neighbor].se && capacity >= q[neighbor]) &&
            (current_time + dist + s[neighbor] + d[0][neighbor])<=timew[0].se )  {
            to_zero = true;
            dfs(current_time + dist, current_dist + dist, capacity-q[neighbor], neighbor);
        }
    }

    double dist = d[0][current_node];

    if (!to_zero)
        dfs(current_time + dist, current_dist + dist, capacity,0);
}

void solve() {
    cin >> n >> Q;

    alpha = 0.5;
    Beta = 0.5;


    if (instance=="1" || 
        instance=="2" ||
        instance=="8" ||
        instance=="13" ||
        instance=="14") {
        alpha = 0.5;
        Beta = 0.5;
    } else if (instance=="7") {
        alpha = 0.91;
        Beta = 0.7;
    } else if (instance=="3" || instance=="9") {
        alpha = 0.6;
        Beta = 0.4;
    } else if (instance=="15") {
        alpha = 0.65;
        Beta = -1;
    }  else if (instance=="4") {
        alpha = 0.65;
        Beta = 0.35;
    }  else if (instance=="10") {
        alpha = 0.75;
        Beta = 0.25;
    }  else if (instance=="16") {
        alpha = 0.87;
        Beta = -1;
    }  else if (instance=="5") {
        alpha = 0.2;
        Beta = 0.2;
    }  else if (instance=="11") {
        alpha = 0.45;
        Beta = -1;
    }  else if (instance=="17") {
        alpha = 0.7;
        Beta = 0.3;
    }  else if (instance=="6") {
        alpha = 0.7;
        Beta =  0.35;
    }  else if (instance=="12") {
        alpha = 0.65;
        Beta = 0.1;
    }  else if (instance=="18") {
        alpha = 0.84;
        Beta = 0;
    }


  
    n++;

    auto start = chrono::high_resolution_clock::now();

    adj.resize(n), coord.resize(n), timew.resize(n), q.resize(n), s.resize(n), visited.resize(n, false), d.resize(n,
        vector<double> (n, 0.0)), dmax.resize(n, 0.0);

    for (int i=0, j; i<n; i++)
        cin >> j >> coord[j].fi >> coord[j].se >> q[j] >> timew[j].fi >> timew[j].se >> s[j];


    for (int i=0; i<n; i++)
        for (int j=i+1; j<n; j++) {
            d[i][j] = distance(coord[i], coord[j]);
            d[j][i] = d[i][j];
            dmax[i] = max(dmax[i], d[i][j]);
        }



    //Podría ser logarítmica
    for (int i=0; i<n; i++) {
        double sal = max(d[0][i],1.0*timew[i].fi) + s[i];

        for (int j=0; j<n; j++) {
            if (i==j) continue;
            double dist = d[i][j];
            double coef = (d[0][i]/dmax[0] * d[0][j]/dmax[0])* (dist - d[j][0]); 

            if ( (sal + dist <= timew[j].se && sal + dist + s[j]+d[0][j]<=timew[0].se && Q-q[i]-q[j]>=0) || j==0) {
                if (i==0)
                    adj[i].pb(make_pair(Beta * dist - (1-Beta) * timew[j].se, j));
                else {
                    if ((instance=="1" || instance=="2" || instance=="7" || instance=="13" || instance=="14") && dist>d[j][0]) continue;
                    adj[i].pb(make_pair(
                        alpha * dist + (1-alpha) * timew[j].fi + coef
                         ,j));
                }
            }
        }
        sort(all(adj[i]));
    }

    for (int j=0; j<n-1; j++)
        adj[0][j].fi -= INF*(sz(adj[adj[0][j].se])==1);
    
    sort(rall(adj[0]));

    /*for (int i=0; i<n; i++) {
        cout << i << ":\n";
        for (auto &[_, node]: adj[i]) {
            double dist = d[i][node];
            cout << _ << " " << node << "\n";
        }
        cout<<"\n\n";
    }*/

    while (total_visited+1<n) {
        fin = false;
        routes.pb({});
        times.pb({});
        dfs(0, 0, Q, 0);
        m++;
        visited[0] = false;
    }

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::milli> duration = end - start;

    cout << m << " " << dist_total << " " << round(duration.count());

    for (int i=0; i<m; i++) {
        cout << "\n" <<sz(routes[i])-2 << " ";
        for (auto &j: routes[i])
            cout << j << " ";
        for (auto &j: times[i])
            cout << j << " ";
        cout << capacities[i];
    }
}

int main(int argc, char* argv[]) {
    ios_base::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);

    string baseDirectory = "C:\\Users\\camilo\\My Drive\\U\\2024-2\\heuristics\\trabajo-1\\VRPTW Instances\\";
    
    instance = argv[2];

    string filePath = baseDirectory + "VRPTW" + instance + ".txt";

    FILE* file = freopen(filePath.c_str(), "r", stdin);

    string output_file_path = argv[1];

    ofstream output_file(output_file_path);
    if (!output_file.is_open()) {
        cerr << "Unable to open file for writing output" << endl;
        return 1;
    }

    streambuf* original_cout_streambuf = cout.rdbuf();
    cout.rdbuf(output_file.rdbuf());

    solve();

    cout.rdbuf(original_cout_streambuf);
    output_file.close();

    return 0;
 }
