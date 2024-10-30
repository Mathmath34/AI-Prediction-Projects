"Temperature prediction as regression problem" In this project we first made an baseline analysis with time series models. Followed by machine learning moddel and approaches to try to figure out the most relevant to predict yhe temperature. The github organization the ramp project organization Here the link to ramp documentation link for more details of the organization

problem.py: https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/problem.html

RAMP starting_kit: https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/advanced/using_kits.html
So what exactly is an ARIMA model?

ARIMA, short for ‘Auto Regressive Integrated Moving Average’ is actually a class of models that ‘explains’ a given time series based on its own past values, that

is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values.

Any ‘non-seasonal’ time series that exhibits patterns and is not a random white noise can be modeled with ARIMA models.

An ARIMA model is characterized by 3 terms: p, d, q

where,

p is the order of the AR term

q is the order of the MA term

d is the number of differencing required to make the time series stationary
What does the p, d and q in ARIMA model mean?

The first step to build an ARIMA model is to make the time series stationary.

Why?

Because, term ‘Auto Regressive’ in ARIMA means it is a linear regression model that uses its own lags as predictors. Linear regression models, as you know, work best when the predictors are not correlated and are independent of each other.

So how to make a series stationary?

The most common approach is to difference it. That is, subtract the previous value from the current value. Sometimes, depending on the complexity of the series, more than one differencing may be needed.

The value of d, therefore, is the minimum number of differencing needed to make the series stationary. And if the time series is already stationary, then d = 0.

Next, what are the ‘p’ and ‘q’ terms?

‘p’ is the order of the ‘Auto Regressive’ (AR) term.

It refers to the number of lags of Y to be used as predictors.

And ‘q’ is the order of the ‘Moving Average’ (MA) term.

It refers to the number of lagged forecast errors that should go into the ARIMA Model.
