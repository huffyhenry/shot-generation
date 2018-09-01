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
    // Team-specific parameters driving shot quality and quantity in log space.
    // The defense vectors are constrained a la Dixon-Coles for identifiability.
    real generation[n_teams];
    real prevention_raw[n_teams-1];
    real conversion[n_teams];
    real obstruction_raw[n_teams-1];

    // Score-dependent modifiers of shooting rate in log space, relative to 0:0.
    // 8 scores from 0:0 to 2:2 (except 0:0) + 3 high score classes.
    real quantity_score_raw[11];

    // Game-state modifiers of shooting rate and shot quality
    real winning_quality;
    real losing_quality;

    // Home advantage modifiers of shooting rate and shot quality
    real hfa_quantity;
    real hfa_quality;

    // Parameters describing dependence of shot rate & quality on waiting time
    real<lower=0.0> quantity_k;
    real quality_k;
}

transformed parameters{
    // Constrained defence ratings
    real prevention[n_teams];
    real obstruction[n_teams];

    // Score modifiers of shot generation rate.
    matrix[3, 3] quantity_score;
    real quantity_winning_other;
    real quantity_drawing_other;
    real quantity_losing_other;

    // Shooting rates and shot quality predictors for all datapoints
    vector[n_shots] production_team;
    vector[n_shots] production_oppo;
    vector[n_shots] shot_quality;


    // Introduce sum-to-zero constraints on team defence coefficients
    prevention[1:(n_teams-1)] = prevention_raw;
    prevention[n_teams] = -sum(prevention_raw);
    obstruction[1:(n_teams-1)] = obstruction_raw;
    obstruction[n_teams] = -sum(obstruction_raw);

    // Rearrange the score modifiers in a matrix,
    // so that quantity_score[x, y] is the modifier for the score x-1:y-1.
    quantity_score[1, 1] = 0.0;
    for (i in 1:3){
      for (j in 1:3){
        if ((i > 1) || (j > 1))
          quantity_score[i, j] = quantity_score_raw[(i - 1)*3 + j - 1];
      }
    }
    quantity_winning_other = quantity_score_raw[9];
    quantity_drawing_other = quantity_score_raw[10];
    quantity_losing_other = quantity_score_raw[11];

    // Compute shooting rates and the shot quality predictor
    for (i in 1:n_shots){
      production_team[i] = generation[team[i]]
                         + prevention[oppo[i]]
                         + home[i]*hfa_quantity;

      production_oppo[i] = generation[oppo[i]]
                         + prevention[team[i]]
                         + (1-home[i])*hfa_quantity;

      if ((scored[i] <= 2) && (conceded[i] <= 2)){
        production_team[i] += quantity_score[scored[i]+1, conceded[i]+1];
        production_oppo[i] += quantity_score[conceded[i]+1, scored[i]+1];
      }
      else if (scored[i] > conceded[i]){
        production_team[i] += quantity_winning_other;
        production_oppo[i] += quantity_losing_other;

      }
      else if (scored[i] < conceded[i]){
        production_team[i] += quantity_losing_other;
        production_oppo[i] += quantity_winning_other;
      }
      else{
        production_team[i] += quantity_drawing_other;
        production_oppo[i] += quantity_drawing_other;
      }

      shot_quality[i] = conversion[team[i]]
                      + obstruction[oppo[i]]
                      + home[i]*hfa_quality
                      + wait[i]*quality_k
                      + (scored[i] > conceded[i] ? winning_quality : 0.0)
                      + (scored[i] < conceded[i] ? losing_quality : 0.0);
    }
}

model{
    // Priors
    generation ~ normal(0, 1);
    prevention_raw ~ normal(0, 1);
    conversion ~ normal(0, 1);
    obstruction_raw ~ normal(0, 1);
    quantity_score_raw ~ normal(0, 1);
    winning_quality ~ normal(0, 1);
    losing_quality ~ normal(0, 1);
    hfa_quantity ~ normal(0, 1);
    hfa_quality ~ normal(0, 1);
    quantity_k ~ normal(1, 0.1);
    quality_k ~ normal(0, 1);

    // Likelihood
    wait ~ weibull(quantity_k, 1.0./exp(production_team));
    goal ~ bernoulli_logit(shot_quality);
    target += weibull_lccdf(wait | quantity_k, 1.0./exp(production_oppo));  // Shot not taken
}

generated quantities{
  // Per-datapoint log-likelihood and nominal number of parameters
  // ONLY FOR THE SHOT GENERATION PART
  real logLik[n_shots];
  int n_params;

  for (i in 1:n_shots)
      logLik[i] = weibull_lpdf(wait[i] | quantity_k, 1.0/exp(production_team[i]))
                + weibull_lccdf(wait[i] | quantity_k, 1.0/exp(production_oppo[i]));

  n_params = 2*n_teams - 1   // generation and prevention skills
           + 1               // HFA
           + 11;             // score classes
}
