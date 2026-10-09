---
title: Underfitting, overfitting, and the U-shaped curve
summary: Watch a polynomial go from too stiff to too wiggly, see training error fall while validation error turns back up, and find out what more data and more noise do to the sweet spot.
lab: fit-explorer
lesson: overfitting-and-bias-variance
minutes: 6
---

# Underfitting, overfitting, and the U-shaped curve

<!-- lab:fit-explorer -->
![Four panels. Degree 1 underfits the sine-shaped data (train MSE 0.275, validation 0.27). Degree 4 follows it (0.075 and 0.09). Degree 15 wiggles wildly (0.056 and 0.52). A validation curve shows training error falling steadily with degree while validation error is lowest around degree 3 to 5 and rises after.](../figures/overfitting-and-bias-variance.png)
<!-- /lab -->

A model can be wrong because it is too simple or because it is too flexible, and training error alone can't tell you which. This explainer turns one knob, polynomial degree, and watches two error curves pull apart. The numbers below come from the lab, which uses 20 training points from $y = \sin(\pi x)$ plus noise.

<!-- step:data -->
## Noisy points from a known curve

The dashed curve is the truth, $y = \sin(\pi x)$. We never get to see it directly. We only get the 20 blue dots, each one the truth plus random noise with standard deviation $\sigma = 0.3$. The teal curve is a polynomial fitted to the dots by least squares. The right panel scores every degree from 1 to 15 twice: the blue line is the error on the 20 training dots, and the orange line is the error on 200 fresh validation points the model never saw. Watch the orange line. It is the only one that tells you how the model will do on new data.

<!-- step:underfit -->
## A straight line can't bend

At degree 1 the model is a straight line, $\hat y = w x + b$. The sine rises and falls, and a line can only tilt, so it misses in the same systematic way everywhere: training error 0.21, validation error 0.30. Both errors are high, and the gap between them is small. That is **underfitting**, also called high **bias**: the average model is wrong, and no amount of extra data fixes it, because the problem is the model's shape, not the sample.

<!-- step:complexity -->
## More bends, less training error

Now raise the degree one step at a time. Each extra coefficient lets the teal curve bend once more, and by degree 3 it already traces the sine: training error drops from 0.21 to 0.034. The blue line on the right only ever goes down. That is guaranteed: a degree-4 polynomial can do anything a degree-3 one can, plus more, so it can never fit the training dots worse. This is why training error is useless for choosing complexity. It always says "more".

<!-- step:overfit -->
## Validation error turns back up

Keep going and something changes on the right. The blue line keeps sliding down, but the orange line, flat from degree 3 to about 9, starts climbing. At degree 12, training error is 0.025 while validation error is 0.57, more than five times its best value. Look at the teal curve: it threads the dots by swinging wildly between and beyond them. It is fitting the noise. That is **overfitting**, or high **variance**: draw a different 20 dots and you would get a completely different curve.

<!-- step:sweet-spot -->
## The bottom of the U

Plot validation error against degree and you get a U. The left wall is bias: too stiff to follow the sine. The right wall is variance: so flexible it follows the noise. For squared error, the two add up, along with noise no model can remove:

```math
\text{expected error} = \text{bias}^2 + \text{variance} + \sigma^2
```

The orange ring marks the bottom: degree 3, validation error 0.10, close to the noise floor $\sigma^2 = 0.09$. That ring, not the lowest training error, is the model to ship.

<!-- step:more-data -->
## More data tames a complex model

Stay at degree 12, the overfit model, and grow the training set from 20 to 60 points. With three times as many dots to pass near, there is less room to swing between them, and the noise in each one averages out. Validation error falls from 0.57 to about 0.13. More data attacks variance, the right wall of the U. It does nothing for bias: a straight line stays at about 0.37 validation error at 60 points, because a line still can't bend.

<!-- step:noise -->
## More noise, more to overfit

Back at 20 points and degree 12, turn the noise down to zero. Now every dot sits exactly on the sine, and degree 12 fits it perfectly: both errors go to zero, and there is no overfitting at all. Now turn $\sigma$ up. The noise is exactly what a flexible model memorizes, so the more of it there is, the wilder the curve: at $\sigma = 0.8$ the degree-12 validation error passes 4. Noisy data calls for simpler models, more data, or regularization.

**Try it yourself:** pick a degree, predict whether going from 20 to 60 training points will help it, then check. Or open the [full lesson](../lessons/02-ml-workflow/02-overfitting-and-bias-variance.md) for the bias-variance math and how to read learning curves.
