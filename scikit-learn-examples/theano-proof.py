__author__ = 'claudio'

import theano

from theano import tensor as T

a = T.scalar()
b = T.scalar()

y = a * b

multiply = theano.function(inputs=[a, b], outputs=y)

print(multiply(1, 3))

print(multiply(4, 6))
