data {
  int<lower=0> N; // Number of observations
  int<lower=0> M; // Number of dyads
  int<lower=0> L; // Number of locations
  int<lower=0> dyad_ids[N]; // Dyad ID corresponding to each social event
  int<lower=0> location_ids[N]; // Location ID corresponding to each social event
  real<lower=0> durations[N]; // Duration of each observed social event (> 0)
  int<lower=0> num_events[M]; // Number of events for each dyad
  vector<lower=0>[M] total_obs_time; // Total amount of time each dyad was observed for
}

parameters {
  vector<lower=0>[M] lambda; // Mean event rates for each dyad
  vector[M] logit_p; // Log social preferences for each dyad
  vector[L] beta_loc; // Parameters for location effects.
  real<lower=0> loc_sigma; // Hyperparameter for location effect adaptive prior standard deviation.
}

transformed parameters {
  vector[N] logit_pn;
  for (i in 1:N) {
    if (location_ids[i] > 0) {
      logit_pn[i] = logit_p[dyad_ids[i]] + beta_loc[location_ids[i]]; // Calculate observation-level social preferences.
    } else {
      logit_pn[i] = logit_p[dyad_ids[i]];
    }
  }
}

model {
  // Main model
  durations ~ exponential(lambda[dyad_ids] ./ inv_logit(logit_pn));
  num_events ~ poisson(lambda .* total_obs_time);

  // Priors
  logit_p ~ normal(0, 2.5);
  lambda ~ normal(0, 0.01);
  beta_loc ~ normal(0, loc_sigma);
  loc_sigma ~ normal(0, 1);
}

generated quantities {
  real event_pred[N] = exponential_rng(lambda[dyad_ids] ./ inv_logit(logit_pn));
}
