## HOW TO RUN

There are 2 files that can be executed:

### main.py
The main.py file is responsible for running the three benchmarks (João Brandão numbers, knapsack, minimum set coverage). Each benchmark will generate files that will be stored inside the 'results' directory.
To run the benchmarks use the following command:

```bash
    python main.py
```

### analise_data.py
The analise_data.py file loads the results files in the 'results' directory and generates the plots that are stored in the 'plots' directory. It also analises the normality of the results and generates plots for visualization.
Then makes the mann whitney statistical test and prints the results.

```bash
    python analise_data.py
```

