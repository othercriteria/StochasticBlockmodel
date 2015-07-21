# Docker deployment

Just putting some working notes here.

The point of the `Dockerfile` here is to instantiate an environment in
which all of the package code will run without problems. This will
expedite testing and deploying to, e.g.,
[Amazon EC2](http://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html)
or other cloud compute resources.

Most of the heavy lifting is done by [`ipython/scipystack`](https://github.com/ipython/docker-notebook).

Will try to keep an up-to-date version of the container at
[`othercriteria/python-network`](https://registry.hub.docker.com/u/othercriteria/python-network/)
on Docker Hub.

TODO: Would be nice to get [`rpy2`](rpy.sourceforge.net/) via `pip`
rather than as the Ubuntu package `python-rpy2`.
