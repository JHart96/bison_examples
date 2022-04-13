data {
  int<lower=0> num_obs; // Number of observations
  int<lower=0> num_dyads; // Number of dyads
  int<lower=0> num_locations; // Number of locations
  int<lower=0> event_count[num_obs]; // Outcome corresponding to each observation (presence/absence)
  int<lower=0> dyad_ids[num_obs]; // Dyad ID corresponding to each data point
  vector<lower=0>[num_obs] durations; // Duration of each observation
  int<lower=0, upper=num_locations> location_ids[num_obs]; // Location ID corresponding to each observation.
}

parameters {
  vector[num_dyads] log_edge; // Logit edge weights for each dyad.
  vector[num_locations] beta_location; // Effects of locations.
  real<lower=0> sigma_loc; // SD for varying intercept of location.
}

transformed parameters {
  vector[num_obs] log_pn = log_edge[dyad_ids] + beta_location[location_ids]; // Logit probability of a social event for each observation.
}

model {
  event_count ~ poisson(exp(log_pn) .* durations);

  // Priors
  beta_location ~ normal(0, sigma_loc);
  sigma_loc ~ normal(0, 1);
  log_edge ~ normal(0, 1);
}

generated quantities {
  int event_pred[num_obs] = poisson_rng(exp(log_pn) .* durations);
}
