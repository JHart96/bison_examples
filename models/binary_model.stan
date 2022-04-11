data {
  int<lower=0> N; // Number of data points
  int<lower=0> M; // Number of dyads
  int<lower=0> L; // Number of locations
  int<lower=0> dyad_ids[N]; // Dyad ID corresponding to each data point
  int<lower=0, upper=1> event[N]; // Outcome corresponding to each data point (presence/absence)
  int<lower=0> location_ids[N]; // Location ID corresponding to each data point
}

parameters {
  vector[M] logit_edge_weight; // Logit edge weights for each dyad.
  vector[L] beta_loc; // Parameters for location effects.
  real<lower=0> loc_sigma; // Hyperparameter for location effect adaptive prior standard deviation.
}

transformed parameters {
  vector[N] logit_pn = logit_edge_weight[dyad_ids] + beta_loc[location_ids] * loc_sigma; // Logit probability of a social event for each observation.
}

model {
  // Main model
  event ~ bernoulli_logit(logit_pn);

  // Adaptive prior over location effects
  beta_loc ~ normal(0, 1);

  // Priors
  logit_edge_weight ~ normal(0, 1);
  loc_sigma ~ normal(0, 1);
}

generated quantities {
  int event_pred[N] = bernoulli_rng(inv_logit(logit_pn));
}
