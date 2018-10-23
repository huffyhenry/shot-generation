# shot-generation
Rigorous Bayesian modelling of shooting in soccer.

### Model
The model is described briefly in a 
[blogpost](https://kwiatkowski.io/shot-generation), for full definition
consult the [shotgen.stan](shotgen.stan) and [shotconv.stan](shotconv.stan)
files.

### Execution
Currently the model is best executed and visualised via IPython, like so:

```python
%run run.py
gen_samples, team_map = run("shotgen.stan")
conv_samples, _ = run("shotconv.stan")

%run analysis.py
team_profiles(gen_samples, conv_samples, team_map)
score_bars(gen_samples)
score_bars(conv_samples)
```
which produces figures like in the [blogpost](https://kwiatkowski.io/shot-generation).
