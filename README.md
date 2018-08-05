# shot-generation
Rigorous Bayesian modelling of shooting in soccer.

### Model
Formal model definition is 
[pending](https://github.com/huffyhenry/shot-generation/issues/25), for now
read the [shotgen.stan](shotgen.stan) file.

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
The model yields a number of interesting insights, but is yet to be 
[checked for overfitting](https://github.com/huffyhenry/shot-generation/issues/4). 
