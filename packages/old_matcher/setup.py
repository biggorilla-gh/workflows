from setuptools import setup

setup(name='flexmatcher',
            version='0.7',
            description='matches schemas from multiple sources to a mediated schema',
            url='http://github.com/biggorilla-gh/biggorilla/packages/flexmatcher',
            author='BigGorilla',
            author_email='thebiggorilla.team@gmail.com',
            packages=['flexmatcher'],
            install_requires=[
                'pandas',
                'scipy'
            ,],
            zip_safe=False)
