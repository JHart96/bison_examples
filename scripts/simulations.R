simulate_binary <- function() {
  ## code to prepare `example.4.2` dataset goes here

  node_names <- c("Rey", "Leia", "Obi-Wan", "Luke", "C-3PO", "BB-8", "R2-D2", "D-O")
  node_types <- c("Lifeform", "Lifeform", "Lifeform", "Lifeform", "Droid", "Droid", "Droid", "Droid")
  location_names <- c("A", "B", "C", "D", "E", "F")

  n <- 8

  node_effects <- rnorm(n, 0, 1)
  logit_p <- matrix(0, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (i < j) {
        logit_p[i, j] <- 5 * (node_types[i] == "Lifeform") * (node_types[j] == "Lifeform") - 5 * (node_types[i] == "Droid") * (node_types[j] == "Droid") + node_effects[i] + node_effects[j] + rnorm(1, 0, 1)
      }
    }
  }
  p <- plogis(logit_p)

  d <- matrix(sample(10:50, size=n^2, replace=TRUE), n, n)
  d <- d * upper.tri(d)

  beta_loc <- rnorm(6, 0, 1)

  df <- data.frame(matrix(nrow=0, ncol=7))
  colnames(df) <- c("node_1", "node_2", "type_1", "type_2", "event", "location", "edge_weight_true")
  for (i in 1:n) {
    for (j in 1:n) {
      if (i < j) {
        for (k in 1:d[i, j]) {
          location_id <- sample(1:6, size=1)
          # At least one of them was visible, did they associate?
          logit_p <- qlogis(p[i, j])
          logit_pn <- logit_p + beta_loc[location_id]
          df[nrow(df) + 1, ] <- c(node_names[i], node_names[j], node_types[i], node_types[j], rbinom(1, 1, plogis(logit_pn)), location_names[location_id], p[i, j])
        }
      }
    }
  }

  df$node_1 <- factor(df$node_1, levels=node_names)
  df$node_2 <- factor(df$node_2, levels=node_names)
  df$type_1 <- factor(df$type_1, levels=c("Lifeform", "Droid"))
  df$type_2 <- factor(df$type_2, levels=c("Lifeform", "Droid"))
  df$location <- factor(df$location, levels=location_names)
  df$event <- as.integer(df$event)
  list(df=df, p=p)
}

# Simulate count data for the INLA example with duration.
simulate_count <- function() {
  ## code to prepare `example.4.2` dataset goes here

  node_names <- c("Rey", "Leia", "Obi-Wan", "Luke", "C-3PO", "BB-8", "R2-D2", "D-O")
  node_types <- c("Lifeform", "Lifeform", "Lifeform", "Lifeform", "Droid", "Droid", "Droid", "Droid")
  location_names <- c("A", "B", "C", "D", "E", "F")

  n <- 8

  node_types_binary <- 1 * (node_types == "Lifeform")
  node_types_binary %*% t(node_types_binary)

  node_effects <- rnorm(n, 0, 1)
  log_p <- matrix(0, n, n)
  for (i in 1:n) {
    for (j in 1:n) {
      if (i < j) {
        log_p[i, j] <- -2 + 2 * (node_types[i] == node_types[j]) + node_effects[i] + node_effects[j] + rnorm(1, 0, 1)
      }
    }
  }

  beta_loc <- rnorm(6, -2, 2)

  d <- matrix(sample(1:50, size=n^2, replace=TRUE), n, n)
  d <- d * upper.tri(d)

  df <- data.frame(matrix(nrow=0, ncol=7))
  colnames(df) <- c("node_1", "node_2", "type_1", "type_2", "event_count", "duration", "location")
  for (i in 1:n) {
    for (j in 1:n) {
      if (i < j) {
        for (k in 1:d[i, j]) {
          duration <- round(runif(1, min=1, max=10))
          location_id <- sample(1:6, size=1)
          # At least one of them was visible, did they associate?
          log_pn <- log_p[i, j] + beta_loc[location_id]
          df[nrow(df) + 1, ] <- c(node_names[i], node_names[j], node_types[i], node_types[j], rpois(1, exp(log_pn) * duration), duration, location_names[location_id])
        }
      }
    }
  }

  df$node_1 <- factor(df$node_1, levels=node_names)
  df$node_2 <- factor(df$node_2, levels=node_names)
  df$type_1 <- factor(df$type_1, levels=c("Lifeform", "Droid"))
  df$type_2 <- factor(df$type_2, levels=c("Lifeform", "Droid"))
  df$event_count <- as.integer(df$event_count)
  df$location <- factor(df$location, levels=location_names)
  list(df=df, p=exp(log_p))
}

simulate_duration <- function() {
  # Define node names and node types
  node_names <- c("Rey", "Leia", "Obi-Wan", "Luke", "C-3PO", "BB-8", "R2-D2", "D-O")
  node_types <- c("Lifeform", "Lifeform", "Lifeform", "Lifeform", "Droid", "Droid", "Droid", "Droid")
  location_names <- c("A", "B", "C", "D", "E", "F")

  # Duration of each sampling/observation period
  obs_time <- 600

  # Create underlying edge weights, rho.
  n <- 8
  logit_p <- matrix(rnorm(n^2, -4, 1), n, n)
  node_types_binary <- 1 * (node_types == "Lifeform")
  logit_p <- logit_p + 3.0 * (node_types_binary %*% t(node_types_binary))
  logit_p <- logit_p * upper.tri(logit_p)

  # Create right-skewed distribution of mean event times, where max_obs_time is the maximum observation time (and therefore maximum event time).
  # mu <- matrix(rbeta(n^2, 20, 100), n, n) * obs_time
  # mu <- mu * upper.tri(mu)

  loc <- rnorm(6)

  lmbd <- matrix(0.001 * rgamma(n^2, 2, 1), n, n)
  lmbd <- lmbd * upper.tri(lmbd)

  # How to set lmbd and mu when there are nuisance effects? Do the maths!
  # Okay, done the maths, now check and implement it!

  d <- matrix(sample(seq(40, 50), n^2, replace=TRUE), n, n) * obs_time
  d <- d * upper.tri(d)

  obs <- data.frame(node_1=character(), node_2=character(), duration=numeric(), event=numeric(), location=character())
  for (i in 1:n) {
    for (j in 1:n) {
      if (i < j) {
        num_events <- rpois(1, lmbd[i, j] * d[i, j])
        for (k in 1:num_events) {
          location_id <- sample(1:6, 1)
          # logit_pn <- rho[i, j] + loc[location_id]
          # pn <- plogis(logit_pn)

          mu <- (1/lmbd[i, j]) * plogis(logit_p[i, j] - loc[location_id])

          duration <- round(min(rexp(1, 1/mu), obs_time)) # Above-truncated by maximum observation time
          obs[nrow(obs) + 1, ] <- list(node_names[i], node_names[j], duration, 1, location_names[location_id])
        }
        if (num_events == 0) {
          obs[nrow(obs) + 1, ] <- list(node_names[i], node_names[j], 0, 0)
        }
      }
    }
  }
  obs$node_1 <- factor(obs$node_1, levels=node_names)
  obs$node_2 <- factor(obs$node_2, levels=node_names)

  obs_agg <- obs %>%
    group_by(node_1, node_2) %>%
    summarise(total_event_time=sum(duration), num_events=sum(event))
  obs_agg$total_obs_time <- t(d)[lower.tri(d)]
  obs_agg$node_1_type <- factor(node_types[obs_agg$node_1], levels=c("Lifeform", "Droid"))
  obs_agg$node_2_type <- factor(node_types[obs_agg$node_2], levels=c("Lifeform", "Droid"))
  obs$location <- factor(obs$location, levels=location_names)
  list(df_obs=obs, df_obs_agg=obs_agg, mu=mu, lmbd=lmbd)
}

simulate_group <- function() {
  obs <- t(sapply(1:50, function(x) rbinom(8, 1, runif(1, min=0.2, max=0.4))))

  df <- data.frame(node_1=numeric(), node_2=numeric(), social_event=numeric(), obs_id=numeric())
  for (obs_id in 1:nrow(obs)) {
    for (i in which(obs[obs_id, ] == 1)) {
      for (j in 1:ncol(obs)) {
        if (i != j) {
          # Swap i and j if necessary to make sure node_1 < node_2, not essential but makes things a bit easier when assigning dyad IDs.
          if (i < j) {
            node_1 <- i
            node_2 <- j
          } else {
            node_1 <- j
            node_2 <- i
          }
          df[nrow(df) + 1, ] <- list(node_1=node_1, node_2=node_2, social_event=(obs[obs_id, i] == obs[obs_id, j]), obs_id=obs_id)
        }
      }
    }
  }
  df
}
