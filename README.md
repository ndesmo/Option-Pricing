Option-Pricing
==============

This is a simple app I designed to price European, Asian and Barrier options in Python using the Monte Carlo method. It can price both calls and puts, as well as all combinations of up, down, in and out for the barriers. For European and Barrier options, the theoretical values are computed, and the graph function allows a graph to be plotted showing the convergence of the price of an option as the number of sample paths increases.

To run the code, you need Python installed, as well as the module NumPy. For the graphs, you also need matplotlib. I would recommend the Enthought Python Distribution as this contains all the modules you need, and is easy to install.

Once you have Python installed, run the program in command prompt or a terminal, typing python pricing.py -h. This will bring up the menu of options, which you can select by replacing the -h as before with any possible combination of options.
