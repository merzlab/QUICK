Hands-on Tutorials
==================

The quick directory contains some test runs and examples. These can be found in the directory examples.
For this tutorial we will use water as an example.  

1. HF energy calculation

::

     HF BASIS=cc-pvDZ CUTOFF=1.0d-10 DENSERMS=1.0d-6 ENERGY

     O                 -0.06756756   -0.31531531    0.00000000
     H                  0.89243244   -0.31531531    0.00000000
     H                 -0.38802215    0.58962052    0.00000000

2. HF gradient calculation

::

     HF BASIS=cc-pvDZ CUTOFF=1.0d-10 DENSERMS=1.0d-6 GRADIENT

     O                 -0.06756756   -0.31531531    0.00000000
     H                  0.89243244   -0.31531531    0.00000000
     H                 -0.38802215    0.58962052    0.00000000    

3. HF geometry optimization calculation

::

     HF BASIS=cc-pvDZ CUTOFF=1.0d-10 DENSERMS=1.0d-6 OPTIMIZE

     O                 -0.06756756   -0.31531531    0.00000000
     H                  0.89243244   -0.31531531    0.00000000
     H                 -0.38802215    0.58962052    0.00000000

