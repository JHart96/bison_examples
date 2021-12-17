data {
  int<lower=0> N; // Number of data points
  int<lower=0> M; // Number of dyads
  int<lower=0> L; // Number of locations
  int<lower=0> dyad_ids[N]; // Dyad ID corresponding to each data point
  int<lower=0> event_count[N]; // Outcome corresponding to each data point (presence/absence)
  int<lower=0> location_ids[N]; // Location ID corresponding to each data point
}

parameters {
  vector[M] log_p; // Logit edge weights for each dyad.
  vector[L] beta_loc; // Parameters for location effects.
  real<lower=0> loc_sigma; // Hyperparameter for location effect adaptive prior standard deviation.
}

transformed parameters {
  vector[N] log_pn = log_p[dyad_ids] + beta_loc[location_ids]; // Logit probability of a social event for each observation.
}

model {
  // Main model
  event_count ~ poisson(exp(log_pn));

  // Adaptive prior over location effects
  beta_loc ~ normal(0, loc_sigma);

  // Priors
  log_p ~ normal(0, 2.5);
  loc_sigma ~ normal(0, 1);
}

generated quantities {
  int event_pred[N] = poisson_rng(exp(log_pn));
}
