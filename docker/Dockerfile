FROM ipython/scipystack:latest

MAINTAINER Daniel Klein <othercriteria@gmail.com>

RUN apt-get update && apt-get install -y \
    libgmp-dev \
    emacs24-nox \
    r-base \
    r-base-dev \
    python-rpy2

RUN pip2.7 install networkx
RUN pip2.7 install xlrd

CMD ["/bin/bash"]
