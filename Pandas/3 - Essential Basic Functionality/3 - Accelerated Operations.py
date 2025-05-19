# Accelerated operations

"""
Pandas has support for accelerating certain types of binary numerical and boolean operations using
the 'numexpr' and 'bottleneck' libraries
These libraries are especially useful when dealing with large datasets, and provide large speedups.

'numexpr' uses smart chunking, caching, and multiple cores
'bottleneck' is a set of specialized cython routines that are especially fast when dealing with arrays
that have nans
"""
