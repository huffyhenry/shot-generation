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

parameters{
    // Team-specific parameters driving shot quality and quantity in log space.
    // The defense vector is constrained a la Dixon-Coles for identifiability.
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
    real production_team[n_shots];
    real production_oppo[n_shots];

    // Linear predictor for conversion
    real shot_quality[n_shots];

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

    // Compute parameter combinations for each shot
    for (i in 1:n_shots){
        // Compute base rates
        production_team[i] = generation[team[i]] + prevention[oppo[i]];
        production_oppo[i] = generation[oppo[i]] + prevention[team[i]];
        shot_quality[i] = conversion[team[i]] + obstruction[oppo[i]] + quality_k*wait[i];

        // Apply home field advantage modifiers
        if (home[i]){
            production_team[i] += hfa_quantity;
            shot_quality[i] += hfa_quality;
        }
        else{
            production_oppo[i] += hfa_quantity;
        }

        // Apply game state modifiers
        if (state[i] > 0){
          production_team[i] += winning_quantity;
          production_oppo[i] += losing_quantity;
          shot_quality[i] += winning_quality;
        }
        else if (state[i] < 0){
          production_team[i] += losing_quantity;
          production_oppo[i] += winning_quantity;
          shot_quality[i] += losing_quality;
        }
    }

    // Likelihood
    wait ~ weibull_lpdf(quantity_k, 1.0./to_vector(exp(production_team)));
    goal ~ bernoulli_logit(shot_quality);
    target += weibull_lccdf(wait | quantity_k, 1.0./to_vector(exp(production_oppo)));
}
