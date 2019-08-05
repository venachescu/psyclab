# Psychophysics and Simulations Laboratory

A collection of Python and C++ extensions to facilitate creating interactive
models and experiments; with the possibility of using the Simulation Laboratory
(SL) library as a base.

Based on glumpy, numpy, scipy, etc.

#### Movement Primitives

The `psyclab.movement.primitives` contains an implementation of the dynamic movement primitive (DMP) framework; these
abstractions make it possible to produce classes of movements that vary in time, starting position or distance but all 
share a central common feature, like converging to a goal location or rhythmic structure.

[1]  Schaal, S. (2006). Dynamic movement primitives-a framework for motor control in humans and humanoid robotics.
Adaptive Motion of Animals and Machines.

#### Muscles Models

### Decision Models

#### Drift Diffusion

#### Neural Fields

[1]

#### Model Pi-Squared

An implementation of policy improvement with path integrals ($\pi$ - squared), a model-free stochastic reinforcement learning method.


[1] Theodorou, E., A generalized path integral control approach to reinforcement learning. Jmlr.org
.
### Interactive Visuals and Audio

### Simulation Laboratory (SL) and Apparatus

The `psyclab.sl` and `psyclab.apparatus` modules provide tools to manage robots simulated with the SL package and
functionality to orchestrate and automate experiments as simulations or with attached to real robotic systems.

[1] Schaal, S. (2009). The SL simulation and real-time control software package. University of Southern California.

## Built with:

* OSCPKT : a minimalistic OSC c++ library 
  Julien Pommier
  
* glumpy : 

## Inspiration from:

* Travis DeWolf's excellent [studywolf blog](https://studywolf.wordpress.com); a great collection of posts going 
into detail about building many similar neuro-motor control simulations.

* Nicolas P. Rougier's implementation of [dynamic neural fields]() and his fantastic `glumpy` library. 

* 