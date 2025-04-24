from setuptools import setup

setup(
    name='sterster',
    version='0.0.1',    
    description='Construct a stellar grid using HEALpy.',
    url='https://github.com/vatsalpanwar/sterster',
    author='Vatsal Panwar',
    author_email='panvatsal@gmail.com',
    license='MIT License',
    packages=['sterster'],
    install_requires=['numpy', 'astropy', 'scipy', 
                      'matplotlib', 'tqdm', 'pyyaml', 'healpy'          
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
)