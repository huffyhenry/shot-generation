data{
    int<lower=2> n_teams;
    int<lower=0> n_shots;

    int<lower=1, upper=n_teams> team[n_shots];
    int<lower=1, upper=n_teams> oppo[n_shots];
    int<lower=0, upper=1> home[n_shots];
    int<lower=0, upper=1> neutral[n_shots];
    int<lower=0, upper=1> shot[n_shots];  // If 0, the point is end of a half
    int<lower=0, upper=1> goal[n_shots];
    int<lower=0, upper=1> own_goal[n_shots];  // Pens and OGs are unmodelled
    int<lower=0, upper=1> penalty[n_shots];
    real<lower=0.0> wait[n_shots];  // Time since last shot or start of the half
    real<lower=0.0> time[n_shots];  // Absolute time of the shot
    int state[n_shots];  // Goal difference at the time of the shot
}

transformed data{
  int<lower=0, upper=1> rebound[n_shots];

  // Define rebound as any shot coming up to 5 seconds after another.
  for (i in 1:n_shots){
    if (wait[i] < 5.0/60.0){
      rebound[i] = 1;
    }
    else{
      rebound[i] = 0;
    }
  }
}

parameters{
    // Team-specific coefficients of shot generation/prevention in log space.
    // Having 2*n_teams - 1 free parameters ensures identifiability.
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

    // Proportion of shots followed by a rebound
    real<lower=0.0, upper=1.0> rebound_rate;
    real rebound_boost;

    // Rate of own goals
    real<lower=0.0> own_goal_rate;

    // Rate of penalties and its HFA modifier, in log space
    real penalty_rate;
    real hfa_penalties;
}

transformed parameters{
    real prevention[n_teams];
    real obstruction[n_teams];

    // Introduce sum-to-zero constraints
    for (i in 1:(n_teams-1)){
        prevention[i] = prevention_raw[i];
        obstruction[i] = obstruction_raw[i];

    }
    prevention[n_teams] = -sum(prevention_raw);
    obstruction[n_teams] = -sum(obstruction_raw);
}

model{
    // Rates of shot production for the two teams, in log space
    real production_team;
    real production_oppo;
    real penalties_team;
    real penalties_oppo;
    real shot_quality;  // An unconstrained measure of conversion probability

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
    quantity_k ~ normal(1, 0.25);
    quality_k ~ normal(0, 1);
    rebound_rate ~ normal(0, 0.25);
    rebound_boost ~ normal(0, 1);
    own_goal_rate ~ normal(0, 0.1);
    penalty_rate ~ normal(0, 10);
    hfa_penalties ~ normal(0, 1);

    // Likelihood
    for (i in 1:n_shots){
        // Compute base rates
        production_team = generation[team[i]] + prevention[oppo[i]];
        production_oppo = generation[oppo[i]] + prevention[team[i]];
        penalties_team = penalty_rate;
        penalties_oppo = penalty_rate;
        shot_quality = conversion[team[i]] + obstruction[oppo[i]] + quality_k*wait[i];

        // Apply home field advantage modifiers
        if (home[i]){
            production_team += hfa_quantity;
            penalties_team += hfa_penalties;
            shot_quality += hfa_quality;
        }
        else if (!neutral[i]){
            production_oppo += hfa_quantity;
            penalties_oppo += hfa_penalties;
        }

        // Apply game state modifiers
        if (state[i] > 0){
          production_team += winning_quantity;
          production_oppo += losing_quantity;
          shot_quality += winning_quality;
        }
        else if (state[i] < 0){
          production_team += losing_quantity;
          production_oppo += winning_quantity;
          shot_quality += losing_quality;
        }

        // Move out of log space
        production_team = exp(production_team);
        production_oppo = exp(production_oppo);
        penalties_team = exp(penalties_team);
        penalties_oppo = exp(penalties_oppo);

        // Add the current datapoint to the likelihood
        if (own_goal[i]){
            wait[i] ~ exponential(own_goal_rate);

            target += bernoulli_lpmf(0 | rebound_rate);
            target += exponential_lccdf(wait[i] | penalties_team);
            target += exponential_lccdf(wait[i] | penalties_oppo);
            target += weibull_lccdf(wait[i] | quantity_k, production_team);
            target += weibull_lccdf(wait[i] | quantity_k, production_oppo);

        }
        else if (penalty[i]){
            wait[i] ~ exponential(penalties_team);

            target += exponential_lccdf(0 | own_goal_rate);
            target += bernoulli_lpmf(0 | rebound_rate);
            target += exponential_lccdf(wait[i] | penalties_oppo);
            target += weibull_lccdf(wait[i] | quantity_k, production_team);
            target += weibull_lccdf(wait[i] | quantity_k, production_oppo);
        }
        else if (rebound[i]){
            // Update the rebound rate only.
            rebound[i] ~ bernoulli(rebound_rate);
            goal[i] ~ bernoulli_logit(shot_quality + rebound_boost);
        }
        else if (shot[i]){
            wait[i] ~ weibull(quantity_k, production_team);
            goal[i] ~ bernoulli_logit(shot_quality);

            target += exponential_lccdf(0 | own_goal_rate);
            target += bernoulli_lpmf(0 | rebound_rate);
            target += exponential_lccdf(wait[i] | penalties_team);
            target += exponential_lccdf(wait[i] | penalties_oppo);
            target += weibull_lccdf(wait[i] | quantity_k, production_oppo);
        }
        else{
            target += exponential_lccdf(0 | own_goal_rate);
            target += bernoulli_lpmf(0 | rebound_rate);
            target += exponential_lccdf(wait[i] | penalties_team);
            target += exponential_lccdf(wait[i] | penalties_oppo);
            target += weibull_lccdf(wait[i] | quantity_k, production_team);
            target += weibull_lccdf(wait[i] | quantity_k, production_oppo);
        }
    }
}
