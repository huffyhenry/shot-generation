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

%run analysis.py
team_scatter(samples, team_map, attack=True)
```
which produces a scatter plot of team attack profiles with 50% CIs:
![Team characteristics](doc/figures/team_scatter.png)

The CIs are very wide, but in fact the underlying team 
coefficients are significantly different from one another, as illustrated
by further running

```python
samples.plot(pars=['generation', 'prevention', 'conversion', 'obstruction'])
plt.show()
```
which yields

![Samples](doc/figures/fit.png)

### Information criteria tracker

To keep an eye on model quality, I record information criteria scores 
of the various variants here. All models are fitted on complete EPL 2017/18
data, using 6 chains x 2000 iterations x 50% warmup = 6000 posterior samples.
For the time being, only the shot generation part of the model is evaluated. 

| Model | Commit | AIC (gen) | WAIC (gen)|
|--------|:------:|---------------:|---------------:|
|[`variant/memoryless`](https://github.com/huffyhenry/shot-generation/tree/variant/memoryless)|[`ac083697`](https://github.com/huffyhenry/shot-generation/commit/ac083697c9904d19da49a807e2733a27bd5d182f)|56533.0|56487.4|
|[`variant/weibull`](https://github.com/huffyhenry/shot-generation/tree/variant/weibull)|[`d0d48093`](https://github.com/huffyhenry/shot-generation/commit/d0d48093a2d58cb14857c03dd176133e3435b332)|**56212.6**|**56172.8**|
|[`variant/weibull+scores`](https://github.com/huffyhenry/shot-generation/tree/variant/weibull%2Bscores)|[`71384c92`](https://github.com/huffyhenry/shot-generation/commit/71384c92b0711c226bfb414604891c24b2382e0a)|56226.7|56178.1|
