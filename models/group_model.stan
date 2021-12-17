data {
  int<lower=0> N; // Number of data points
  int<lower=0> M; // Number of dyads
  int<lower=0> G; // Number of groups
  int<lower=0> dyad_ids[N]; // Dyad ID corresponding to each data point
  int<lower=0, upper=1> event[N]; // Outcome corresponding to each data point (presence/absence)
  int<lower=0> group_ids[N]; // Location ID corresponding to each data point
}

parameters {
  vector[M] logit_p; // Logit edge weights for each dyad.
  vector[G] beta_group; // Parameters for group effects.
  real<lower=0> sigma_group; // Hyperparameter for location effect adaptive prior standard deviation.
}

transformed parameters {
  vector[N] logit_pn = logit_p[dyad_ids] + beta_group[group_ids]; // Logit probability of a social event for each observation.
}

model {
  // Main model
  event ~ bernoulli(inv_logit(logit_pn));

  // Adaptive prior over location effects
  beta_group ~ normal(0, sigma_group);

  // Priors
  logit_p ~ normal(0, 1);
  sigma_group ~ normal(0, 1);
}

generated quantities {
  int event_pred[N] = bernoulli_rng(inv_logit(logit_pn));
}
