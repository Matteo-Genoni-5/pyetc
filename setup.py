from setuptools import setup, find_packages

setup(
    name='pyetc',
    version='0.3',
    description='Python ETC for instrument simulations',
    packages=find_packages(),  # détecte automatiquement les sous-modules dans pyetc/
    include_package_data=True,  # permet d’inclure les fichiers non .py listés dans MANIFEST.in
    zip_safe=False,  # nécessaire si tu accèdes à des fichiers avec open() ou os.path.join()
    install_requires=[
        'astropy',
        'numpy',
        # ajoute d'autres dépendances si nécessaire
    ],
)