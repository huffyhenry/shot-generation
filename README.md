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
graph(samples, team_map, attack=True)
```
which produces a scatter plot of team attack profile with 50% CIs:
![Team coefficients](doc/figures/team_scatter.png)

The CIs are very wide, but in fact the underlying team 
coefficients are significantly different from one another, as illustrated
by further running

```python
samples.plot(pars=['generation', 'prevention', 'conversion', 'obstruction'])
plt.show()
```
which yields

![Samples](doc/figures/fit.png)

### Status
The model is unfit for (any) purpose until a number of improvements are made as per
the [Issues](https://github.com/huffyhenry/shot-generation/issues?q=is%3Aissue+is%3Aopen+label%3Amodel-design). 
