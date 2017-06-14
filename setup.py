from distutils.core import setup, Extension
import numpy

mod = Extension('DemNets',
    include_dirs = [numpy.get_include()],
    sources = ['DemNets.c'],
    #extra_compile_args=['-ggdb', '-fopenmp'],#, '-O0'],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-lgomp']
)

setup (name = 'DemNets',
    author = 'Aljoscha Rheinwalt',
    author_email = 'aljoscha.rheinwalt@uni-potsdam.de',
    ext_modules = [mod]
)
