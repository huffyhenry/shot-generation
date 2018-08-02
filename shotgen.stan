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

parameters{
    // Team-specific coefficients of shot generation and prevention
    // Having 2*n_teams - 1 free params ensures identifiability, see below.
    real<lower=0.0> generation[n_teams];
    real<lower=0.0> prevention_raw[n_teams-1];

    // Game-state modifiers of shooting rate, relative to that at drawn state
    real<lower=0.0> winning;
    real<lower=0.0> losing;

    real<lower=0.0> hfa;
}

transformed parameters{
    real<lower=0.0> prevention[n_teams];

    // Introduce multiply-to-one constraint.
    for (i in 1:(n_teams-1)){
        prevention[i] = prevention_raw[i];
    }
    prevention[n_teams] = exp(-sum(log(prevention_raw)));
}

model{
    // Rates of shot production for the two teams
    real production_team;
    real production_oppo;

    // Priors
    generation ~ normal(0, 1);
    prevention_raw ~ normal(0, 1);  // _raw b/c the constraint is non-linear
    winning ~ normal(1, 1);
    losing ~ normal(1, 1);
    hfa ~ normal(1, 1);

    // Likelihood
    for (i in 1:n_shots){
        // Compute the rates of shot production
        production_team = generation[team[i]] * prevention[oppo[i]];
        production_oppo = generation[oppo[i]] * prevention[team[i]];
        if (home[i]){
            production_team *= hfa;
        }
        else if (!neutral[i]){
            production_oppo *= hfa;
        }
        if (state[i] > 0){
          production_team *= winning;
          production_oppo *= losing;
        }
        else if (state[i] < 0){
          production_team *= losing;
          production_oppo *= winning;
        }

        // Add the current datapoint to the likelihood
        if (wait[i] > 0.0){  // 0.0 suggests a rebound, unmodelled for now
            if (own_goal[i] || penalty[i] || !shot[i]){
                // No shot taken for a period of time
                target += exponential_lccdf(wait[i] | production_team);
                target += exponential_lccdf(wait[i] | production_oppo);
            }
            else {
                // One team takes a shot, meaning that the other does not
                target += exponential_lpdf(wait[i] | production_team);
                target += exponential_lccdf(wait[i] | production_oppo);
            }
        }
    }
}
