// An example of a pooled BISoN count model accounting for visibilites due to location.
// Partial pooling is applied to all edge weights by applying a shared adaptive prior over `log_p`.
// If edge weights can be considered as belonging to an underlying univariate distribution, this partial
// pooling can increase precision and reduce uncertainty around edge weight estimates.

// Partial pooling can also be applied conditionally based on known dyad types, for example if some dyads
// are known to be, say, siblings, they may share an adaptive prior, whereas unrelated dyads may share a different
// adaptive prior.

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
  real mu_log_p; // Pooling effect for mean log_p.
  real<lower=0> sigma_log_p; // Pooling effect for sd log_p.
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

  // Partial pooling
  log_p ~ normal(mu_log_p, sigma_log_p);

  // Priors
  mu_log_p ~ normal(0, 1);
  sigma_log_p ~ normal(0, 1);
  loc_sigma ~ normal(0, 1);
}

generated quantities {
  int event_pred[N] = poisson_rng(exp(log_pn));
}
