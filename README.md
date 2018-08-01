# shot-generation
Rigorous Bayesian modelling of shooting in soccer.

### Model
There is a dedicated wiki page with 
[Model definition](https://github.com/huffyhenry/shot-generation/wiki).

### Execution
Currently the model is best executed and visualised via IPython, like so:
```python
%run run.py
samples, team_map = run()
graph(samples, team_map)
```
which produces a scatter plot of team coefficients with 50% CIs:
![Team coefficients](doc/figures/team_scatter.png)

### Status
The model is unfit for (any) purpose until a number of improvements are made as per
the [Issues](https://github.com/huffyhenry/shot-generation/issues?q=is%3Aissue+is%3Aopen+label%3Amodel-design). 
