data{
    int<lower=2> n_teams;
    int<lower=0> n_shots;

    int<lower=1, upper=n_teams> team[n_shots];
    int<lower=1, upper=n_teams> oppo[n_shots];
    int<lower=0, upper=1> home[n_shots];
    int<lower=0, upper=1> goal[n_shots];
    real<lower=0.0> wait[n_shots];  // Time since last shot or start of the half
    real<lower=0.0> time[n_shots];  // Absolute time of the shot

    // Exact score at the time of shot
    int<lower=0> scored[n_shots];
    int<lower=0> conceded[n_shots];
}

parameters{
    // Team-specific parameters driving shot quantity in log space.
    // The obstruction vector is constrained a la Dixon-Coles for identifiability.
    real conversion[n_teams];
    real obstruction_raw[n_teams-1];

    // Score-dependent modifiers of shooting rate in log space, relative to 0:0.
    // 8 scores from 0:0 to 2:2 (except 0:0) + 3 high score classes.
    real score_raw[11];

    // Home advantage
    real hfa;
}

transformed parameters{
    real obstruction[n_teams];

    matrix[3, 3] score;
    real winning_other;
    real drawing_other;
    real losing_other;

    vector[n_shots] shot_quality;

    // Introduce sum-to-zero constraints on team defence coefficients
    obstruction[1:(n_teams-1)] = obstruction_raw;
    obstruction[n_teams] = -sum(obstruction_raw);

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
    drawing_other = score_raw[10];
    losing_other = score_raw[11];

    // Compute shot quality vector
    for (i in 1:n_shots){
      shot_quality[i] = conversion[team[i]] + obstruction[oppo[i]] + home[i]*hfa;

      if ((scored[i] <= 2) && (conceded[i] <= 2)){
        shot_quality[i] += score[scored[i]+1, conceded[i]+1];
      }
      else if (scored[i] > conceded[i]){
        shot_quality[i] += winning_other;

      }
      else if (scored[i] < conceded[i]){
        shot_quality[i] += losing_other;
      }
      else{
        shot_quality[i] += drawing_other;
      }
    }
}

model{
    // Priors
    conversion ~ normal(0, 1);
    obstruction_raw ~ normal(0, 1);
    score_raw ~ normal(0, 1);
    hfa ~ normal(0, 1);

    // Likelihood
    goal ~ bernoulli_logit(shot_quality);
}

generated quantities{
  // Per-datapoint log-likelihood and nominal number of parameters.
  real logLik[n_shots];
  int n_params;

  for (i in 1:n_shots)
      logLik[i] = bernoulli_logit_lpmf(goal[i] | shot_quality[i]);

  n_params = 2*n_teams - 1   // conversion and obstruction skills
           + 1               // HFA
           + 11;             // score classes
}
