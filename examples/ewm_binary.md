Binary Edge Weight Model
================

This example covers fitting an edge weight model to presence/absence
(binary) data with an observation-level location effect.

*Note: Many of the procedures presented here are stochastic, and plots
and results may vary between compilations of this document. In
particular, MCMC chains and model estimates may sometimes not be
optimal.*

# Setup

First of all we’ll load in Rstan for model fitting in Stan, dplyr for
handling the data, and igraph for network plotting and computing network
centrality. We also load in a custom R file: “simulations.R” to generate
synthetic data for this example..

``` r
library(rstan)
library(dplyr)
library(igraph)

source("../scripts/simulations.R")
```

# Simulating data

Now we will simulate data using the `simulate_binary()` function. The
rows of the resulting dataframe describe observations at the dyadic
level between nodes. In this dataframe, `event` denotes whether or not
an undirected social event was observed in this observation period. The
exact definition of observation period will depend on the study, but is
commonly a sampling period where at least one of the members of the dyad
was observed. This can also be a sampling period where both members of
the dyad were observed, and the distinction will affect the
interpretation of edge weights. See the paper for further discussion on
this. `location` denotes the location at which the observation took
place, which may be relevant if location is likely to impact the
visibility of social events.

``` r
set.seed(123)
data <- simulate_binary()
df <- data$df
head(df)
```

    ##   node_1 node_2   type_1   type_2 event location
    ## 1    Rey   Leia Lifeform Lifeform     1        E
    ## 2    Rey   Leia Lifeform Lifeform     1        C
    ## 3    Rey   Leia Lifeform Lifeform     1        D
    ## 4    Rey   Leia Lifeform Lifeform     1        D
    ## 5    Rey   Leia Lifeform Lifeform     1        C
    ## 6    Rey   Leia Lifeform Lifeform     1        F

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
  mutate(location_id=as.integer(location))
head(df)
```

    ## # A tibble: 6 × 8
    ## # Groups:   node_1, node_2 [1]
    ##   node_1 node_2 type_1   type_2   event location dyad_id location_id
    ##   <fct>  <fct>  <fct>    <fct>    <int> <fct>      <int>       <int>
    ## 1 Rey    Leia   Lifeform Lifeform     1 E              1           5
    ## 2 Rey    Leia   Lifeform Lifeform     1 C              1           3
    ## 3 Rey    Leia   Lifeform Lifeform     1 D              1           4
    ## 4 Rey    Leia   Lifeform Lifeform     1 D              1           4
    ## 5 Rey    Leia   Lifeform Lifeform     1 C              1           3
    ## 6 Rey    Leia   Lifeform Lifeform     1 F              1           6

``` r
df_agg <- df %>%
  group_by(node_1, node_2) %>%
  summarise(event_count=sum(event), dyad_id=cur_group_id(), total_obs=n()) %>%
  mutate(node_1_id=as.integer(node_1), node_2_id=as.integer(node_2)) %>%
  mutate(sri=event_count/total_obs)
head(df_agg)
```

    ## # A tibble: 6 × 8
    ## # Groups:   node_1 [1]
    ##   node_1 node_2  event_count dyad_id total_obs node_1_id node_2_id   sri
    ##   <fct>  <fct>         <int>   <int>     <int>     <int>     <int> <dbl>
    ## 1 Rey    Leia             44       1        50         1         2 0.88 
    ## 2 Rey    Obi-Wan          20       2        31         1         3 0.645
    ## 3 Rey    Luke             28       3        34         1         4 0.824
    ## 4 Rey    C-3PO             4       4        39         1         5 0.103
    ## 5 Rey    BB-8              0       5        16         1         6 0    
    ## 6 Rey    R2-D2             0       6        43         1         7 0

Now we have all of the data in the right format for fitting the model,
we just need to put it into a list object. The data required by the
statistical model is defined in `binary_model.stan`.

``` r
model_data <- list(
  N=nrow(df), # Number of observations
  M=nrow(df_agg), # Number of dyads
  L=6, # Number of locations
  dyad_ids=df$dyad_id, # Vector of dyad IDs corresponding to each observation
  location_ids=df$location_id, # Vector of location IDs corresponding to each observation
  event=df$event # Vector of binary values (0/1, presence/absence) corresponding to each observation
)
model_data
```

    ## $N
    ## [1] 856
    ## 
    ## $M
    ## [1] 28
    ## 
    ## $L
    ## [1] 6
    ## 
    ## $dyad_ids
    ##   [1]  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
    ##  [26]  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
    ##  [51]  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2
    ##  [76]  2  2  2  2  2  2  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3
    ## [101]  3  3  3  3  3  3  3  3  3  3  3  3  3  3  3  4  4  4  4  4  4  4  4  4  4
    ## [126]  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4  4
    ## [151]  4  4  4  4  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  5  6  6  6  6  6
    ## [176]  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6  6
    ## [201]  6  6  6  6  6  6  6  6  6  6  6  6  6  7  7  7  7  7  7  7  7  7  7  7  7
    ## [226]  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7  7
    ## [251]  7  7  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8
    ## [276]  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8  8
    ## [301]  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9  9
    ## [326]  9  9  9  9  9  9 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10
    ## [351] 10 10 11 11 11 11 11 11 11 11 11 11 11 11 12 12 12 12 12 12 12 12 12 12 12
    ## [376] 12 12 12 12 12 12 12 12 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13
    ## [401] 13 13 13 13 13 13 13 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14
    ## [426] 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 15
    ## [451] 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15
    ## [476] 15 15 15 15 15 15 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16
    ## [501] 16 16 16 16 16 16 16 16 16 16 16 16 16 17 17 17 17 17 17 17 17 17 17 17 17
    ## [526] 17 17 17 17 17 17 17 17 17 17 17 17 17 17 17 17 17 17 17 18 18 18 18 18 18
    ## [551] 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18
    ## [576] 18 18 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19
    ## [601] 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 20 20 20 20 20 20 20 20 20
    ## [626] 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 21 21 21 21 21 21 21 21 21 21
    ## [651] 21 21 21 21 21 21 21 21 21 21 21 22 22 22 22 22 22 22 22 22 22 22 22 22 22
    ## [676] 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 23 23 23 23 23 23 23
    ## [701] 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 24 24
    ## [726] 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24
    ## [751] 24 24 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 25 26 26 26 26 26 26 26
    ## [776] 26 26 26 26 26 26 26 26 26 26 26 26 26 26 26 26 26 26 26 27 27 27 27 27 27
    ## [801] 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27 27
    ## [826] 27 27 27 27 27 27 27 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28
    ## [851] 28 28 28 28 28 28
    ## 
    ## $location_ids
    ##   [1] 5 3 4 4 3 6 1 2 5 4 4 6 4 6 4 3 4 4 1 3 3 6 5 3 6 1 4 2 5 6 5 2 2 5 2 4 6
    ##  [38] 2 1 5 3 5 5 3 4 1 2 4 1 1 6 6 1 2 2 5 3 4 2 4 2 5 5 4 1 1 4 1 5 1 5 2 6 5
    ##  [75] 1 5 5 5 4 5 1 1 5 1 3 2 4 5 5 2 6 3 3 3 3 2 3 5 6 6 4 6 4 4 5 5 3 3 2 1 4
    ## [112] 3 1 6 3 2 1 2 2 2 4 4 4 1 2 2 3 3 4 3 5 6 6 2 5 2 4 4 4 2 3 6 2 3 3 1 1 3
    ## [149] 4 2 4 2 4 1 3 2 5 6 2 3 6 4 3 3 4 4 6 6 2 3 6 4 1 3 1 4 4 5 4 5 6 3 2 4 4
    ## [186] 1 3 6 2 5 1 6 6 6 2 3 1 3 3 5 6 4 4 1 4 4 5 4 6 6 3 3 1 5 4 5 3 4 5 3 6 6
    ## [223] 3 3 3 6 2 2 6 5 5 6 3 5 1 2 6 4 5 1 3 1 2 5 3 5 1 5 4 1 5 2 6 1 6 6 2 1 1
    ## [260] 4 3 2 1 5 4 4 4 4 3 4 6 1 1 3 3 3 2 6 4 1 5 4 5 5 4 1 4 3 2 4 6 3 2 5 4 3
    ## [297] 5 1 1 5 4 6 4 6 1 3 3 1 2 4 1 1 3 4 5 6 3 2 5 5 1 2 3 2 3 3 2 4 6 4 1 2 2
    ## [334] 3 6 3 3 1 6 5 5 1 1 3 2 3 1 4 6 2 2 3 2 4 2 6 1 6 5 6 2 4 3 6 2 1 2 4 2 5
    ## [371] 1 3 2 4 4 5 4 2 2 4 5 3 2 1 5 3 5 2 3 6 5 4 3 2 2 2 2 3 6 1 6 1 3 2 4 4 4
    ## [408] 6 3 3 4 1 4 1 6 2 4 1 2 3 6 4 6 2 6 4 2 2 1 5 1 4 3 6 5 1 3 5 6 1 4 5 3 4
    ## [445] 5 4 6 3 6 3 1 4 6 4 1 2 5 3 3 2 4 1 5 5 6 5 1 5 2 1 2 6 5 2 2 6 4 2 6 6 3
    ## [482] 3 6 2 4 3 4 1 3 2 5 6 2 2 2 4 1 1 3 3 5 6 5 3 6 3 1 3 6 5 2 5 3 3 6 5 2 3
    ## [519] 5 1 1 3 6 6 1 5 1 1 6 6 6 6 5 4 5 5 6 1 2 5 5 4 6 4 1 1 6 6 4 5 2 4 6 1 6
    ## [556] 4 1 6 3 4 5 1 5 4 6 2 3 4 5 1 3 2 1 3 2 3 5 6 1 6 2 5 4 1 5 6 5 6 6 1 6 2
    ## [593] 6 5 2 6 3 3 4 1 6 1 1 5 6 2 2 2 5 3 5 3 1 4 6 1 5 4 1 4 1 4 3 4 3 3 4 3 2
    ## [630] 4 4 2 6 4 4 5 3 1 3 4 5 5 1 4 3 3 1 6 3 6 1 1 2 6 2 4 3 3 6 2 4 1 5 5 1 6
    ## [667] 1 4 2 1 1 2 4 3 1 2 3 3 2 3 3 2 5 4 6 1 5 5 3 6 6 5 2 5 6 1 2 5 5 3 3 6 6
    ## [704] 1 4 6 3 1 6 3 1 5 6 1 5 6 3 6 5 4 4 4 4 2 3 3 5 2 3 2 6 4 2 4 1 5 5 1 1 5
    ## [741] 1 6 3 1 3 2 5 5 5 1 3 1 1 6 2 2 1 5 2 2 1 1 6 4 1 6 3 6 5 3 1 6 3 3 1 3 1
    ## [778] 1 2 2 3 2 5 6 5 1 2 3 2 1 6 2 6 5 5 1 6 3 4 3 6 1 6 4 5 5 2 6 6 5 3 1 2 5
    ## [815] 1 2 2 6 5 6 2 1 3 4 1 2 6 3 6 6 4 2 2 3 4 3 3 2 4 3 1 5 6 2 6 1 6 3 3 6 4
    ## [852] 3 1 4 5 4
    ## 
    ## $event
    ##   [1] 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1
    ##  [38] 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0 1 1 1 1 1 0 1 0 0 1 1 0 1 0 1 0 1 1 1 1
    ##  [75] 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 0 0 1 1 1 0 1 1 1 1 0 1 1 1 1 1 1 1 1 0 1
    ## [112] 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0
    ## [149] 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ## [186] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ## [223] 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1
    ## [260] 1 1 1 1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 1 1 1 0 0 1
    ## [297] 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 0
    ## [334] 0 1 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ## [371] 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 1
    ## [408] 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 0 1 0 1 0
    ## [445] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ## [482] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
    ## [519] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
    ## [556] 0 1 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0
    ## [593] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
    ## [630] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 1 0
    ## [667] 0 0 1 0 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ## [704] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0
    ## [741] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
    ## [778] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    ## [815] 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0
    ## [852] 0 1 1 0 0

# Fitting the model

To fit the model, we first must compile it and load it into memory using
the function `stan_model()` and providing the filepath to the model. The
working directory will need to be set to the directory of the model for
this to work properly.

``` r
model <- stan_model("../models/ewm_binary.stan")
```

Compiling the model may take a minute or two, but once this is done, the
model can be fit using `sampling()`. The argument `cores` sets the
number of CPU cores to be used for fitting the model, if your computer
has 4 or more cores, it’s worth setting this to 4.

``` r
fit <- sampling(model, model_data, cores=4)
```

# Model checking

The R-hat values provided by Stan indicate how well the chains have
converged, with values very close to 1.00 being ideal. Values diverging
from 1.00 indicate that the posterior samples may be very unreliable,
and shouldn’t be trusted. The chains can be plotted using Rstan’s
`traceplot` function to verify this visually:

``` r
traceplot(fit)
```

![](ewm_binary_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

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
event_pred <- extract(fit)$event_pred
num_iterations <- dim(event_pred)[1]

# Plot the density of the observed event counts
plot(density(df_agg$event_count), main="", xlab="Dyadic event counts")

# Plot the densities of the predicted event counts, repeat for 10 samples
df_copy <- df
for (i in 1:20) {
  df_copy$event <- event_pred[sample(1:num_iterations, size=1), ]
  df_agg_copy <- df_copy %>% 
    group_by(node_1, node_2) %>%
    summarise(event_count=sum(event))
  lines(density(df_agg_copy$event_count), col=rgb(0, 0, 1, 0.5))
}
```

![](ewm_binary_files/figure-gfm/unnamed-chunk-9-1.png)<!-- --> This plot
shows that the observed data falls well within the predicted densities,
and the predictions suggest the model has captured the main features of
the data well. Now we can be reasonably confident that the model has fit
correctly and describes the data well, so we can start to make
inferences from the model.

# Extracting edge weights

The main purpose of this part of the framework is to estimate edge
weights of dyads. We can access these using the `logit_p` quantity. This
will give a distribution of logit-scale edge weights for each dyad, akin
to an edge list. We’ll apply the logistic function `plogis` to get the
edge weights back to their original scale:

``` r
logit_p_samples <- plogis(extract(fit)$logit_p)
```

We can summarise the distribution over edge lists by calculating the
credible intervals, indicating likely values for each edge. We’ll use
the 89% credible interval in this example, but there’s no reason to
choose this interval over any other. The distribution over edge lists
can be summarised in the following code:

``` r
dyad_name <- do.call(paste, c(df_agg[c("node_1", "node_2")], sep=" <-> "))
edge_lower <- apply(logit_p_samples, 2, function(x) quantile(x, probs=0.055))
edge_upper <- apply(logit_p_samples, 2, function(x) quantile(x, probs=0.945))
edge_median <- apply(logit_p_samples, 2, function(x) quantile(x, probs=0.5))
edge_list <- cbind(
  "median"=round(edge_median, 3), 
  "5.5%"=round(edge_lower, 3), 
  "94.5%"=round(edge_upper, 3)
)
rownames(edge_list) <- dyad_name
edge_list
```

    ##                   median  5.5% 94.5%
    ## Rey <-> Leia       0.954 0.917 0.977
    ## Rey <-> Obi-Wan    0.868 0.771 0.927
    ## Rey <-> Luke       0.926 0.866 0.961
    ## Rey <-> C-3PO      0.326 0.173 0.510
    ## Rey <-> BB-8       0.219 0.077 0.449
    ## Rey <-> R2-D2      0.134 0.050 0.285
    ## Rey <-> D-O        0.190 0.078 0.364
    ## Leia <-> Obi-Wan   0.939 0.892 0.968
    ## Leia <-> Luke      0.937 0.882 0.969
    ## Leia <-> C-3PO     0.390 0.203 0.601
    ## Leia <-> BB-8      0.258 0.089 0.516
    ## Leia <-> R2-D2     0.286 0.120 0.502
    ## Leia <-> D-O       0.545 0.352 0.707
    ## Obi-Wan <-> Luke   0.946 0.895 0.974
    ## Obi-Wan <-> C-3PO  0.191 0.082 0.361
    ## Obi-Wan <-> BB-8   0.205 0.083 0.395
    ## Obi-Wan <-> R2-D2  0.229 0.093 0.428
    ## Obi-Wan <-> D-O    0.424 0.249 0.596
    ## Luke <-> C-3PO     0.294 0.148 0.484
    ## Luke <-> BB-8      0.252 0.105 0.461
    ## Luke <-> R2-D2     0.399 0.214 0.600
    ## Luke <-> D-O       0.464 0.280 0.642
    ## C-3PO <-> BB-8     0.167 0.061 0.347
    ## C-3PO <-> R2-D2    0.335 0.171 0.520
    ## C-3PO <-> D-O      0.390 0.186 0.622
    ## BB-8 <-> R2-D2     0.228 0.092 0.432
    ## BB-8 <-> D-O       0.196 0.081 0.372
    ## R2-D2 <-> D-O      0.430 0.243 0.625

In social network analysis, a more useful format for network data is
usually adjacency matrices, rather than edge lists, so now we’ll convert
the distribution of edge lists to a distribution of adjacency matrices,
and store the result in an 8 x 8 x 4000 tensor, as there are 8 nodes and
4000 samples from the posterior.

``` r
adj_tensor <- array(0, c(8, 8, num_iterations))
for (dyad_id in 1:model_data$M) {
  dyad_row <- df_agg[df_agg$dyad_id == dyad_id, ]
  adj_tensor[dyad_row$node_1_id, dyad_row$node_2_id, ] <- logit_p_samples[, dyad_id]
}
adj_tensor[, , 1] # Print the first sample of the posterior distribution over adjacency matrices
```

    ##      [,1]      [,2]      [,3]      [,4]      [,5]      [,6]       [,7]
    ## [1,]    0 0.9770396 0.9342114 0.9355612 0.3155411 0.3117205 0.19150177
    ## [2,]    0 0.0000000 0.9486209 0.9590870 0.3928452 0.3405488 0.09663996
    ## [3,]    0 0.0000000 0.0000000 0.9550397 0.2087090 0.1233833 0.13179334
    ## [4,]    0 0.0000000 0.0000000 0.0000000 0.4595660 0.2528542 0.56964800
    ## [5,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.2989669 0.19628704
    ## [6,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.33763889
    ## [7,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.00000000
    ## [8,]    0 0.0000000 0.0000000 0.0000000 0.0000000 0.0000000 0.00000000
    ##           [,8]
    ## [1,] 0.4182777
    ## [2,] 0.5880662
    ## [3,] 0.5693102
    ## [4,] 0.5009655
    ## [5,] 0.1919905
    ## [6,] 0.1524374
    ## [7,] 0.4058339
    ## [8,] 0.0000000

The adjacency matrix above corresponds to a single draw of the posterior
adjacency matrices. You’ll notice the edges have been transformed back
to the \[0, 1\] range from the logit scale using the logistic function.
If there are no additional effects (such as location in our case), the
transformed edge weights will be probabilities and the median will be
approximately the same as the simple ratio index for each dyad. However,
when additional effects are included, the transformed values can no
longer be interpreted as probabilities, though they will be useful for
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

# Calculate width of credible intervals.
adj_range <- adj_upper - adj_lower
adj_range[is.nan(adj_range)] <- 0

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

![](ewm_binary_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

This plot can be extended in multiple ways, for example by thresholding
low edge weights to visualise the network more tidily, or by adding
halos around nodes to show uncertainty around node centrality, and so
on.

# Next Steps

Now the edge weight model has been fitted, the edge weight posteriors
can be used in the various types of network analyses shown in this
repository.
