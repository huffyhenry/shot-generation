data{
    int<lower=2> n_teams;
    int<lower=0> n_shots;

    int<lower=1, upper=n_teams> team[n_shots];
    int<lower=1, upper=n_teams> oppo[n_shots];
    int<lower=0, upper=1> home[n_shots];
    int<lower=0, upper=1> goal[n_shots];
    real<lower=0.0> wait[n_shots];  // Time since last shot or start of the half
    real<lower=0.0> time[n_shots];  // Absolute time of the shot
    int state[n_shots];  // Goal difference at the time of the shot
}

transformed data{
  // Game state recoded to facilitate vectorization
  vector[n_shots] winning;
  vector[n_shots] losing;

  for (i in 1:n_shots){
    winning[i] = 0;
    losing[i] = 0;
    if (state[i] > 0)
      winning[i] = 1;
    else if (state[i] < 0)
      losing[i] = 1;
  }
}

parameters{
    // Team-specific parameters driving shot quality and quantity in log space.
    // The defense vectors are constrained a la Dixon-Coles for identifiability.
    real generation[n_teams];
    real prevention_raw[n_teams-1];
    real conversion[n_teams];
    real obstruction_raw[n_teams-1];

    // Game-state modifiers of shooting rate and shot quality
    real winning_quantity;
    real winning_quality;
    real losing_quantity;
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

    // Shooting rates and shot quality predictors for all datapoints
    vector[n_shots] production_team;
    vector[n_shots] production_oppo;
    vector[n_shots] shot_quality;

    // Introduce sum-to-zero constraints on team defence coefficients
    prevention[1:(n_teams-1)] = prevention_raw;
    prevention[n_teams] = -sum(prevention_raw);
    obstruction[1:(n_teams-1)] = obstruction_raw;
    obstruction[n_teams] = -sum(obstruction_raw);

    // Compute shooting rates and the shot quality predictor
    production_team = exp(
      to_vector(generation[team])
      + to_vector(prevention[oppo])
      + to_vector(home)*hfa_quantity
      + winning*winning_quantity
      + losing*losing_quantity
    );
    production_oppo = exp(
      to_vector(generation[oppo])
      + to_vector(prevention[team])
      + (1 - to_vector(home))*hfa_quantity
      + losing*winning_quantity
      + winning*losing_quantity
    );
    shot_quality = to_vector(conversion[team])
                   + to_vector(obstruction[oppo])
                   + to_vector(home)*hfa_quality
                   + to_vector(wait)*quality_k
                   + winning*winning_quality
                   + losing*losing_quality;
}

model{
    // Priors
    generation ~ normal(0, 1);
    prevention ~ normal(0, 1);
    conversion ~ normal(0, 1);
    obstruction ~ normal(0, 1);
    winning_quantity ~ normal(0, 1);
    winning_quality ~ normal(0, 1);
    losing_quantity ~ normal(0, 1);
    losing_quality ~ normal(0, 1);
    hfa_quantity ~ normal(0, 1);
    hfa_quality ~ normal(0, 1);
    quantity_k ~ normal(1, 0.1);
    quality_k ~ normal(0, 1);

    // Likelihood
    wait ~ weibull(quantity_k, 1.0./production_team);
    goal ~ bernoulli_logit(shot_quality);
    target += weibull_lccdf(wait | quantity_k, 1.0./production_oppo);  // Shot not taken
}
