data {
  int<lower=0> N; // Number of data points
  int<lower=0> M; // Number of dyads
  int<lower=0> dyad_ids[N]; // Dyad ID corresponding to each data point
  int<lower=0> event_count[N]; // Outcome corresponding to each data point (presence/absence)
  vector<lower=0>[N] durations; // Duration of observation corresponding to each data point.
}

parameters {
  vector[M] log_p; // Logit edge weights for each dyad.
}

transformed parameters {
  vector[N] log_pn = log_p[dyad_ids]; // Logit probability of a social event for each observation.
}

model {
  event_count ~ poisson(exp(log_pn) .* durations);

  // Priors
  log_p ~ normal(1, 0.75);
}

generated quantities {
  int event_pred[N] = poisson_rng(exp(log_pn) .* durations);
}
