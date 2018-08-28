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

| Commit | Model | AIC (gen only) | WAIC (gen only)|
|--------|-------|---------------:|---------------:|
|[`2ba71d73`](https://github.com/huffyhenry/shot-generation/commit/2ba71d73081e558819acc1ca92ba7a7cefb9514c)|Exponential|56617.0|56487.5|
|[`ae5298a5`](https://github.com/huffyhenry/shot-generation/commit/ae5298a579228c24ac637bd5236db5618de0e5e8)|Weibull base|**56298.6**|**56172.7**|
