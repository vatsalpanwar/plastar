from setuptools import setup

setup(
    name='plastar',
    version='0.0.1',    
    description='Computing high-resolution time-series of stellar spectrum with a transiting or eclipsing planet. Built upon spotter that uses the HEALPix subdivision scheme.',
    url='https://github.com/vatsalpanwar/plastar',
    author='Vatsal Panwar',
    author_email='panvatsal@gmail.com',
    license='MIT License',
    packages=['sterster'],
    install_requires=['numpy', 'astropy', 'scipy', 
                      'matplotlib', 'tqdm', 'pyyaml', 'healpy', 'spotter'          
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)