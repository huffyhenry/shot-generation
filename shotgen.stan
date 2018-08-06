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

    // Parameters describing dependence of shot rate & quality on waiting time.
    real<lower=0.0> quantity_k;
    real quality_k;

    // Proportion of shots followed by a rebound
    real<lower=0.0, upper=1.0> rebound_rate;
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

    // Likelihood
    for (i in 1:n_shots){
        production_team = generation[team[i]] + prevention[oppo[i]];
        production_oppo = generation[oppo[i]] + prevention[team[i]];
        shot_quality = conversion[team[i]] + obstruction[oppo[i]] + quality_k*wait[i];
        if (home[i]){
            production_team += hfa_quantity;
            shot_quality += hfa_quality;
        }
        else if (!neutral[i]){
            production_oppo += hfa_quantity;
        }
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

        // Add the current datapoint to the likelihood
        if (own_goal[i] || penalty[i] || !shot[i]){
            // No shot taken for a period of time
            target += weibull_lccdf(wait[i] | quantity_k, exp(production_team));
            target += weibull_lccdf(wait[i] | quantity_k, exp(production_oppo));
        }
        else if (rebound[i]){
            // The shot is a rebound. Update rebound rate only.
            target += bernoulli_lpmf(1 | rebound_rate);
        }
        else{
            // One team takes a (regular) shot, the other does not.
            target += bernoulli_lpmf(0 | rebound_rate);
            target += weibull_lpdf(wait[i] | quantity_k, exp(production_team));
            target += weibull_lccdf(wait[i] | quantity_k, exp(production_oppo));
            target += bernoulli_logit_lpmf(goal[i] | shot_quality);
        }
    }
}
