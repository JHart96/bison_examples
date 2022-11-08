data {
  int<lower=0> num_obs; // Number of observations
  int<lower=0> num_dyads; // Number of dyads
  int<lower=0> num_locations; // Number of locations
  int<lower=0> dyad_ids[num_obs]; // Dyad ID corresponding to each social event
  int<lower=0> location_ids[num_obs]; // Location ID corresponding to each social event
  real<lower=0> durations[num_obs]; // Duration of each observed social event (> 0)
  int<lower=0> num_events[num_dyads]; // Number of events for each dyad
  vector<lower=0>[num_dyads] total_obs_time; // Total amount of time each dyad was observed for
}

parameters {
  vector<lower=0>[num_dyads] lambda; // Mean event rates for each dyad
  vector[num_dyads] logit_edge; // Log edge weights for each dyad
  vector[num_locations] beta_loc; // Parameters for location effects.
  real<lower=0> loc_sigma; // Hyperparameter for location effect adaptive prior standard deviation.
}

transformed parameters {
  vector[num_obs] logit_pn;
  for (i in 1:num_obs) {
    if (location_ids[i] > 0) {
      logit_pn[i] = logit_edge[dyad_ids[i]] + beta_loc[location_ids[i]]; // Calculate observation-level edge weights.
    } else {
      logit_pn[i] = logit_edge[dyad_ids[i]];
    }
  }
}

model {
  // Main model
  durations ~ exponential(lambda[dyad_ids] ./ inv_logit(logit_pn)); // T[, max_duration]; // Uncomment to truncate the exponential.
  num_events ~ poisson(lambda .* total_obs_time);

  // Priors
  logit_edge ~ normal(0, 2.5);
  lambda ~ normal(0, 0.01);
  beta_loc ~ normal(0, loc_sigma);
  loc_sigma ~ normal(0, 1);
}

generated quantities {
  real event_pred[num_obs] = exponential_rng(lambda[dyad_ids] ./ inv_logit(logit_pn));
}
