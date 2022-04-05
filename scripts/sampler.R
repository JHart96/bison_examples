metropolis <- function(target, initial, iterations=10000, warmup=2000, thin=4, chain_id=1, refresh=2000) {
  k <- length(initial)
  chain <- matrix(0, iterations + warmup, k)

  ll_obs <- NULL
  pred <- NULL

  chain[1, ] <- initial

  acceptances <- c(1)

  prop_dispersion <- 2.38

  for (i in 2:(iterations + warmup)) {
    current <- chain[i - 1, ]
    # Single Component Adaptive Metropolis (SCAM)

    if (i <= 100) {
      candidate <- rnorm(k, mean=current, sd=5)
    } else {
      prop_var <- prop_dispersion * (current_var + 0.05)
      candidate <- rnorm(k, mean=current, sd=sqrt(prop_var))
    }

    current_lk <- target(current, i)
    candidate_lk <- target(candidate, i)

    A <- exp(candidate_lk - current_lk)

    if (runif(1) < A) { # Accept
      chain[i, ] <- candidate
      acceptances[length(acceptances) + 1] <- 1
    } else {
      acceptances[length(acceptances) + 1] <- 0
      chain[i, ] <- current
    }

    # Update current mean and current var for next iteration.
    if (i == 100) {
      current_mean <- colMeans(chain[1:i, ])
      current_var <- apply(chain[1:i, ], 2, var)
    } else if (i >= 100) {
      prev_mean <- current_mean
      prev_var <- current_var

      current_mean <- (i - 1)/i * prev_mean + (1/i) * chain[i, ]
      current_var <- (i - 2)/(i - 1) * prev_var + prev_mean^2 + 1/(i - 1) * (chain[i, ])^2 - i/(i - 1) * current_mean^2
    }

    # Adjust acceptance rate for batch.
    if (i %% 100 == 0) {
      acc_batch <- mean(acceptances[(i - 99):i])
      if (acc_batch < 0.23) {
        prop_dispersion <- prop_dispersion/exp(sqrt(1/(i/100)))
      } else {
        prop_dispersion <- prop_dispersion * exp(sqrt(1/(i/100)))
      }
    }

    # Print out progress
    if (refresh != 0 && i %% refresh == 0) {
      if (i < warmup) {
        cat(paste0("Chain: ", chain_id, " | Iteration: ", i, "/", warmup + iterations, " (Warmup)\n"))
      } else {
        cat(paste0("Chain: ", chain_id, " | Iteration: ", i, "/", warmup + iterations, " (Sampling)\n"))
      }
    }
  }

  # close(pb)
  cat(paste0("Acceptance Rate: ", mean(acceptances), "\n"))
  return(chain[seq(warmup, iterations + warmup - 1, thin), ])
}
