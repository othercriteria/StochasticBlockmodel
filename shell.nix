{ pkgs ? import <nixpkgs> {} }:

with pkgs;
python27Packages.buildPythonApplication {
  name = "ArrayStats";

  buildInputs = ((with pkgs;
     [
		   freetype
       gmp
       graphviz
		   libpng
       pcre
		   pkgconfig
       R
       readline
		  ]) ++
    (with python27Packages;
		 [
		   pip
		   setuptools
		   virtualenv

       ipython
		   jupyter

       graphviz
		   matplotlib
       networkx
		   numpy
		   pandas
       pydot
       (scikitlearn.overridePythonAttrs (oldAttrs: { checkPhase = "true"; }))
		   scipy
		   seaborn
       statsmodels
		 ]));

  shellHook = ''
    echo 'Entering ArrayStats dev environment'

    set -v
    alias pip="PIP_PREFIX='$(pwd)/pip_packages' pip"
    export PYTHONPATH="$(pwd)/pip_packages/lib/python2.7/site-packages:$PYTHONPATH"
    export PATH="$(pwd)/pip_packages/bin:$PATH"
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.readline ]}
    set +v
  '';
}
