Here's a minimally cleaned-up dump of notes on the procedure for staging experiments on AWS.

As a reminder, to detach from a running Docker container, use <kbd>Ctrl</kbd>+<kbd>P</kbd><kbd>Ctrl</kbd>+<kbd>Q</kbd> (see [this Stack Overflow thread](http://stackoverflow.com/questions/20145717/how-to-detach-from-a-docker-container) for details).

## Setup of new instance

Make sure `empirical-trials-2015-08-05.pem` or equivalent exists.

Navigate to `StochasticBlockmodel/docker/`

`chmod 400 empirical-trials-2015-08-05.pem`

`ssh -i “empirical-trials-2015-08-05.pem” ec2-user@PUBLIC-DNS`

`sudo yum update`

`sudo yum install -y docker`

`sudo service docker start`

`sudo usermod -a -G docker ec2-user`

disconnect/reconnect

`docker info`

`docker login`

`docker run -v /home/ec2-user/runs:/tmp/StochasticBlockmodel/runs -it othercriteria/python-network`

`cd StochasticBlockmodel`

`git clone https://github.com/othercriteria/StochasticBlockmodel.git`

`cp StochasticBlockmodel/*`

TODO: fix this directory structure mess.

`./build.sh`

Verify `build_params.py`

Verify `test.py`

## Interacting with S3

`rsync -avzh -e "ssh -i docker/empirical-trials-2015-08-05.pem" ec2-user@PUBLIC-DNS:/home/ec2-user/runs/*completed.json runs/`

`aws s3 sync runs/ s3://daniel-klein-data/experiment-1/`

`aws s3 ls s3://daniel-klein-data/experiment-2/ | grep completed | awk '{print $NF}' | sort -V`
