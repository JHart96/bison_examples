Duration Data Example
================

This example covers fitting a edge weight model to duration data (where
the duration of social events is recorded) with an observation-level
location effect, basic model checking and diagnostics, visualising
networks with uncertainty, calculating probability distributions over
network centrality, and propagating network uncertainty into subsequent
analyses.

*Note: Many of the procedures presented here are stochastic, and plots
and results may vary between compilations of this document. In
particular, MCMC chains and model estimates may sometimes not be
optimal, even if they were when it was originally written.*

# Setup

First of all we’ll load in Rstan for model fitting in Stan, dplyr for
handling the data, and igraph for network plotting and computing network
centrality. We also load in two custom R files: “simulations.R” to
generate synthetic data for this example; and “sampler.R” to allow
fitting models with uncertainty over network features respectively.

``` r
library(rstan)
library(dplyr)
library(igraph)

source("../scripts/simulations.R")
source("../scripts/sampler.R")
```

# Simulating data

Now we will simulate data using the `simulate_duration()` function. The
rows of the resulting dataframe describe observations at the dyadic
level between nodes. In this dataframe, `event` denotes whether or not
an undirected social event was observed in this observation period, and
`duration` denotes the length of such an event. The exact definition of
observation period will depend on the study, but is commonly a sampling
period where at least one of the members of the dyad was observed. This
can also be a sampling period where both members of the dyad were
observed, and the distinction will affect the interpretation of edge
weights. See the paper for further discussion on this. `location`
denotes the location at which the observation took place, which may be
relevant if location is likely to impact the visibility of social
events.

``` r
set.seed(1)
data <- simulate_duration()
```

    ## `summarise()` has grouped output by 'node_1'. You can override using the
    ## `.groups` argument.

``` r
df <- data$df_obs
df_agg <- data$df_obs_agg
head(df)
```

    ##   node_1 node_2 duration event location
    ## 1    Rey   Leia      600     1        A
    ## 2    Rey   Leia       42     1        A
    ## 3    Rey   Leia       35     1        D
    ## 4    Rey   Leia       68     1        E
    ## 5    Rey   Leia        3     1        B
    ## 6    Rey   Leia       59     1        D

``` r
head(df_agg)
```

    ## # A tibble: 6 × 7
    ## # Groups:   node_1 [1]
    ##   node_1 node_2  total_event_time num_events total_obs_time node_1_type
    ##   <fct>  <fct>              <dbl>      <dbl>          <dbl> <fct>      
    ## 1 Rey    Leia               12833        114          28800 Lifeform   
    ## 2 Rey    Obi-Wan             5620         87          28200 Lifeform   
    ## 3 Rey    Luke                7788         66          26400 Lifeform   
    ## 4 Rey    C-3PO               1174         51          24000 Lifeform   
    ## 5 Rey    BB-8                 699         59          30000 Lifeform   
    ## 6 Rey    R2-D2                860         13          25800 Lifeform   
    ## # … with 1 more variable: node_2_type <fct>

In our simulated data, `total_event_time` is the total duration of
social events, `num_events` is the number of social events, and
`total_obs_time` is the total duration of observations.

# Preparing the data

Computationally it’s easier to work with dyad IDs rather than pairs of
nodes in the statistical model, so we’ll map the pairs of nodes to dyad
IDs before we put the data into the model. The same is true for the
location factor, so we will also map the locations to location IDs. We
can add these columns to the dataframe using the following code:

``` r
df <- df %>%
  group_by(node_1, node_2) %>%
  mutate(dyad_id=cur_group_id()) %>%
  mutate(location_id=as.integer(location), location_id=ifelse(is.na(location_id), 0, location_id))
head(df)
```

    ## # A tibble: 6 × 7
    ## # Groups:   node_1, node_2 [1]
    ##   node_1 node_2 duration event location dyad_id location_id
    ##   <fct>  <fct>     <dbl> <dbl> <fct>      <int>       <int>
    ## 1 Rey    Leia        600     1 A              1           1
    ## 2 Rey    Leia         42     1 A              1           1
    ## 3 Rey    Leia         35     1 D              1           4
    ## 4 Rey    Leia         68     1 E              1           5
    ## 5 Rey    Leia          3     1 B              1           2
    ## 6 Rey    Leia         59     1 D              1           4

It will also be useful later to aggregate the dataframe at the dyad
level and assign dyad IDs corresponding to each dyad. We can do this
using:

``` r
df_agg <- df_agg %>%
  group_by(node_1, node_2) %>%
  mutate(dyad_id=cur_group_id(), node_1_id=as.integer(node_1), node_2_id=as.integer(node_2))
head(df_agg)
```

    ## # A tibble: 6 × 10
    ## # Groups:   node_1, node_2 [6]
    ##   node_1 node_2  total_event_time num_events total_obs_time node_1_type
    ##   <fct>  <fct>              <dbl>      <dbl>          <dbl> <fct>      
    ## 1 Rey    Leia               12833        114          28800 Lifeform   
    ## 2 Rey    Obi-Wan             5620         87          28200 Lifeform   
    ## 3 Rey    Luke                7788         66          26400 Lifeform   
    ## 4 Rey    C-3PO               1174         51          24000 Lifeform   
    ## 5 Rey    BB-8                 699         59          30000 Lifeform   
    ## 6 Rey    R2-D2                860         13          25800 Lifeform   
    ## # … with 4 more variables: node_2_type <fct>, dyad_id <int>, node_1_id <int>,
    ## #   node_2_id <int>

Now we have all of the data in the right format for fitting the model,
we just need to put it into a list object. The data required by the
statistical model is defined in `duration_model.stan`.

``` r
model_data <- list(
  num_obs=nrow(df), # Number of observations
  num_dyads=nrow(df_agg), # Number of dyads
  num_locations=6, # Number of locations
  dyad_ids=df$dyad_id, # Vector of dyad IDs corresponding to each observation
  location_ids=df$location_id, # Vector of location IDs corresponding to each observation
  durations=df$duration, # Vector of event durations corresponding to each observation
  num_events=df_agg$num_events, # Vector of event counts corresponding to each dyad
  total_obs_time=df_agg$total_event_time # Vector of total observation times corresponding to each dyad
)
```

# Fitting the model

To fit the model, we first must compile it and load it into memory using
the function `stan_model()` and providing the filepath to the model. The
working directory will need to be set to the directory of the model for
this to work properly.

``` r
edge_model <- stan_model("../models/edge_duration.stan")
```

Compiling the model may take a minute or two, but once this is done, the
model can be fit using `sampling()`. The argument `cores` sets the
number of CPU cores to be used for fitting the model, if your computer
has 4 or more cores, it’s worth setting this to 4.

``` r
fit_edge <- sampling(edge_model, model_data, cores=4)
```

    ## Warning: Bulk Effective Samples Size (ESS) is too low, indicating posterior means and medians may be unreliable.
    ## Running the chains for more iterations may help. See
    ## https://mc-stan.org/misc/warnings.html#bulk-ess

# Model checking

The R-hat values provided by Stan indicate how well the chains have
converged, with values very close to 1.00 being ideal. Values diverging
from 1.00 indicate that the posterior samples may be very unreliable,
and shouldn’t be trusted. The chains can be plotted using Rstan’s
`traceplot` function to verify this visually:

``` r
traceplot(fit_edge)
```

![](ewm_duration_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

Good R-hat values don’t necessarily indicate that the model is
performing well, only that the parameter estimates appear to be robust.
To check that the model is performing as it should, a predictive check
can be used. A predictive check uses the fitted model to make
predictions, and compares those predictions to the observed data. The
predictions should indicate that the observed data are concordant with
the predictions from the model. There are many ways to perform a
predictive check, as data can be summarised in many different ways. For
the purposes of this example, we’ll use a simple density check where the
probability distributions of the aggregated event counts are compared
against the predictions from the model. Note that this isn’t a guarantee
that the model predictions are good, only that the predictions have the
same event count distribution as the data. Ideally several predictive
checks would be used to check the performance of the model.

This check uses predictions generated by the Stan model as the quantity
`event_pred`, with one set of predictions for each step in the MCMC
chain. The predictive check will randomly sample 10 of these steps,
compute the event counts for each dyad, and plot the densities against
the density of the observed event counts from the data.

``` r
# Extract event predictions from the fitted model
event_pred <- extract(fit_edge)$event_pred
num_iterations <- dim(event_pred)[1]

# Plot the density of the observed event counts
plot(density(df$duration), main="", xlab="Dyadic event durations", ylim=c(0, 0.04))

# Plot the densities of the predicted event counts, repeat for 10 samples
df_copy <- df
for (i in 1:20) {
  event_durations_pred <- event_pred[sample(1:num_iterations, size=1), ]
  lines(density(event_durations_pred), col=rgb(0, 0, 1, 0.5))
}
```

![](ewm_duration_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

This plot shows that the observed data falls well within the predicted
densities, and the predictions suggest the model has captured the main
features of the data well. Now we can be reasonably confident that the
model has fit correctly and describes the data well, so we can start to
make inferences from the model.

# Extracting edge weights

The main purpose of this part of the framework is to estimate edge
weights of dyads. We can access these using the `logit_p` quantity. This
will give a distribution of logit-scale edge weights for each dyad, akin
to an edge list. A more useful format for network data is usually
adjacency matrices, rather than edge lists, so instead we’ll convert the
distribution of edge lists to a distribution of adjacency matrices, and
store the result in an 8 x 8 x 4000 tensor, as there are 8 nodes and
4000 samples from the posterior.

``` r
logit_edge_samples <- extract(fit_edge)$logit_edge
edge_samples <- plogis(logit_edge_samples)

dyad_name <- do.call(paste, c(df_agg[c("node_1", "node_2")], sep=" <-> "))
edge_lower <- apply(edge_samples, 2, function(x) quantile(x, probs=0.025))
edge_upper <- apply(edge_samples, 2, function(x) quantile(x, probs=0.975))
edge_median <- apply(edge_samples, 2, function(x) quantile(x, probs=0.5))
edge_list <- cbind(
  "median"=round(edge_median, 3), 
  "2.5%"=round(edge_lower, 3), 
  "97.5%"=round(edge_upper, 3)
)
rownames(edge_list) <- dyad_name
edge_list
```

    ##                   median  2.5% 97.5%
    ## Rey <-> Leia       0.799 0.600 0.914
    ## Rey <-> Obi-Wan    0.787 0.589 0.907
    ## Rey <-> Luke       0.853 0.668 0.955
    ## Rey <-> C-3PO      0.440 0.225 0.689
    ## Rey <-> BB-8       0.340 0.172 0.579
    ## Rey <-> R2-D2      0.459 0.143 0.974
    ## Rey <-> D-O        0.317 0.151 0.551
    ## Leia <-> Obi-Wan   0.892 0.573 0.996
    ## Leia <-> Luke      0.670 0.439 0.848
    ## Leia <-> C-3PO     0.270 0.134 0.477
    ## Leia <-> BB-8      0.359 0.174 0.613
    ## Leia <-> R2-D2     0.652 0.377 0.865
    ## Leia <-> D-O       0.185 0.083 0.372
    ## Obi-Wan <-> Luke   0.777 0.509 0.952
    ## Obi-Wan <-> C-3PO  0.340 0.148 0.628
    ## Obi-Wan <-> BB-8   0.363 0.172 0.609
    ## Obi-Wan <-> R2-D2  0.421 0.206 0.671
    ## Obi-Wan <-> D-O    0.563 0.304 0.814
    ## Luke <-> C-3PO     0.297 0.138 0.535
    ## Luke <-> BB-8      0.369 0.193 0.593
    ## Luke <-> R2-D2     0.246 0.116 0.449
    ## Luke <-> D-O       0.722 0.455 0.916
    ## C-3PO <-> BB-8     0.264 0.111 0.540
    ## C-3PO <-> R2-D2    0.686 0.329 0.950
    ## C-3PO <-> D-O      0.750 0.477 0.914
    ## BB-8 <-> R2-D2     0.446 0.212 0.742
    ## BB-8 <-> D-O       0.252 0.125 0.449
    ## R2-D2 <-> D-O      0.608 0.360 0.826

``` r
logit_adj_tensor <- array(0, c(8, 8, num_iterations))
adj_tensor <- array(0, c(8, 8, num_iterations))
for (dyad_id in 1:model_data$num_dyads) {
  dyad_row <- df_agg[df_agg$dyad_id == dyad_id, ]
  logit_adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- logit_edge_samples[, dyad_id]
  adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- edge_samples[, dyad_id]
}
adj_tensor[, , 1] # Print the first sample of the posterior distribution over adjacency matrices
```

    ##      [,1]     [,2]      [,3]      [,4]      [,5]      [,6]      [,7]      [,8]
    ## [1,]    0 0.813437 0.8373489 0.9399784 0.4733107 0.2533872 0.2947485 0.3204074
    ## [2,]    0 0.000000 0.7868288 0.6661396 0.2402268 0.2857090 0.7459528 0.1623009
    ## [3,]    0 0.000000 0.0000000 0.7931429 0.4346603 0.4816170 0.4771647 0.4938675
    ## [4,]    0 0.000000 0.0000000 0.0000000 0.2360686 0.4348068 0.2097984 0.5635529
    ## [5,]    0 0.000000 0.0000000 0.0000000 0.0000000 0.2284331 0.3852175 0.7206582
    ## [6,]    0 0.000000 0.0000000 0.0000000 0.0000000 0.0000000 0.2643985 0.1997740
    ## [7,]    0 0.000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.3946621
    ## [8,]    0 0.000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000

The adjacency matrix above corresponds to a single draw of the posterior
adjacency matrices. You’ll notice the edges have been transformed back
to the \[0, 1\] range from the logit scale using the logistic function.
If there are no additional effects (such as location in our case), the
transformed edge weights will be proportions and the median will be
approximately the same as the simple ratio index for each dyad. However,
when additional effects are included, the transformed values can no
longer be interpreted as proportions, though they will be useful for
visualisation and analysis purposes.

# Visualising uncertainty

The aim of our network visualisation is to plot a network where the
certainty in edge weights (edge weights) can be seen. To do this we’ll
use a semi-transparent line around each edge with a width that
corresponds to a standardised uncertainty measures. The uncertainty
measure will simply be the normalised difference between the 97.5% and
2.5% credible interval estimate for each edge weight. We can calculate
this from the transformed adjacency tensor object, generate two igraph
objects for the main network and the uncertainty in edges, and plot them
with the same coordinates.

``` r
# Calculate lower, median, and upper quantiles of edge weights. Lower and upper give credible intervals.
adj_quantiles <- apply(adj_tensor, c(1, 2), function(x) quantile(x, probs=c(0.025, 0.5, 0.975)))
adj_lower <- adj_quantiles[1, , ]
adj_mid <- adj_quantiles[2, , ]
adj_upper <- adj_quantiles[3, , ]

# Calculate standardised width/range of credible intervals.
adj_range <- adj_upper - adj_lower

# Generate two igraph objects, one form the median and one from the standardised width.
g_mid <- graph_from_adjacency_matrix(adj_mid, mode="undirected", weighted=TRUE)
g_range <- graph_from_adjacency_matrix(adj_range, mode="undirected", weighted=TRUE)

# Plot the median graph first and then the standardised width graph to show uncertainty over edges.
coords <- igraph::layout_nicely(g_mid)
plot(g_mid, edge.width=3 * E(g_mid)$weight, edge.color="black",  layout=coords)
plot(g_mid, edge.width=20 * E(g_range)$weight, edge.color=rgb(0, 0, 0, 0.25), 
     vertex.label=c("Rey", "Leia", "Obi-Wan", "Luke", "C-3PO", "BB-8", "R2-D2", "D-O"), 
     vertex.label.dist=4, vertex.label.color="black", layout=coords, add=TRUE)
```

![](ewm_duration_files/figure-gfm/unnamed-chunk-12-1.png)<!-- --> This
plot can be extended in multiple ways, for example by thresholding low
edge weights to visualise the network more tidily, or by adding halos
around nodes to show uncertainty around network centrality, and so on.

# Next Steps

Now the edge weight model has been fitted, the edge weight posteriors
can be used in the various types of network analyses shown in this
repository.
