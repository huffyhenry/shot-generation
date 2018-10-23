data{
    int<lower=2> n_teams;
    int<lower=0> n_shots;

    int<lower=1, upper=n_teams> team[n_shots];
    int<lower=1, upper=n_teams> oppo[n_shots];
    int<lower=0, upper=1> home[n_shots];
    real<lower=0.0> wait[n_shots];  // Time since last shot or start of the half
    real<lower=0.0> time[n_shots];  // Absolute time of the shot

    // Exact score at the time of shot
    int<lower=0> scored[n_shots];
    int<lower=0> conceded[n_shots];
}

parameters{
    // Team-specific parameters driving shot quantity in log space.
    // Constrained a la Dixon-Coles for identifiability.
    real generation[n_teams];
    real prevention_raw[n_teams-1];

    // Score-dependent modifiers of shooting rate in log space, relative to 0:0.
    // 8 scores from 0:0 to 2:2 (except 0:0) + 3 high score classes.
    real score_raw[11];

    // Home advantage
    real hfa;

    // Effect of absolute time in the game
    real t_raw;

    // The Weibull shape parameter, kept away from 0 to help the sampler.
    real<lower=0.1> k;
}

transformed parameters{
    real prevention[n_teams];

    matrix[3, 3] score;
    real winning_other;
    real drawing_other;
    real losing_other;

    real t;

    vector[n_shots] production_team;
    vector[n_shots] production_oppo;

    // Scale the absolute time dependence to help the sampler
    t = t_raw / 1000;

    // Introduce sum-to-zero constraints on team defence coefficients
    prevention[1:(n_teams-1)] = prevention_raw;
    prevention[n_teams] = -sum(prevention_raw);

    // Rearrange the score modifiers in a matrix,
    // so that score[x, y] is the modifier for the score x-1:y-1.
    score[1, 1] = 0.0;
    for (i in 1:3){
      for (j in 1:3){
        if ((i > 1) || (j > 1))
          score[i, j] = score_raw[(i - 1)*3 + j - 1];
      }
    }
    winning_other = score_raw[9];
    losing_other = score_raw[10];
    drawing_other = score_raw[11];

    // Compute shooting rates at datapoint level
    for (i in 1:n_shots){
      production_team[i] = generation[team[i]]
                         + prevention[oppo[i]]
                         + hfa*home[i]
                         + t*(time[i] - wait[i]/2);
      production_oppo[i] = generation[oppo[i]]
                         + prevention[team[i]]
                         + (1-home[i])*hfa
                         + t*(time[i] - wait[i]/2);

      if ((scored[i] <= 2) && (conceded[i] <= 2)){
        production_team[i] += score[scored[i]+1, conceded[i]+1];
        production_oppo[i] += score[conceded[i]+1, scored[i]+1];
      }
      else if (scored[i] > conceded[i]){
        production_team[i] += winning_other;
        production_oppo[i] += losing_other;

      }
      else if (scored[i] < conceded[i]){
        production_team[i] += losing_other;
        production_oppo[i] += winning_other;
      }
      else{
        production_team[i] += drawing_other;
        production_oppo[i] += drawing_other;
      }
    }
}

model{
    // Priors
    generation ~ normal(0, 10);
    prevention_raw ~ normal(0, 10);
    score_raw ~ normal(0, 1);
    hfa ~ normal(0, 1);
    t_raw ~ normal(0, 10);
    k ~ normal(1, 1);

    // Likelihood
    wait ~ weibull(k, 1.0./exp(production_team));
    target += weibull_lccdf(wait | k, 1.0./exp(production_oppo));  // Shot not taken
}

generated quantities{
  // Per-datapoint log-likelihood and nominal number of parameters.
  real logLik[n_shots];
  int n_params;

  for (i in 1:n_shots)
      logLik[i] = weibull_lpdf(wait[i] | k, 1.0/exp(production_team[i]))
                + weibull_lccdf(wait[i] | k, 1.0/exp(production_oppo[i]));

  n_params = 2*n_teams - 1   // generation and prevention skills
           + 1               // Weibull shape
           + 2               // time dependence parameters
           + 1               // HFA
           + 11;             // score classes
}
