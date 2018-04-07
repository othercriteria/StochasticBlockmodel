{ pkgs ? import <nixpkgs> {} }:

with pkgs;
python27Packages.buildPythonApplication {
  name = "ArrayStats";

  buildInputs = ((with pkgs;
     [
		   freetype
       gmp
		   libpng
		   pkgconfig
		  ]) ++
    (with python27Packages;
		 [
		   pip
		   setuptools
		   virtualenv

		   jupyter
		   matplotlib
		   numpy
		   pandas
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
    set +v
  '';
}
