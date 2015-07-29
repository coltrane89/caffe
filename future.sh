#!/bin/bash
git checkout master
git branch -D future
git checkout -b future
## merge PRs
# coord maps, net pointer, crop layer # THIS NEEDS TO BE FIXEDDDDDD
hub merge https://github.com/BVLC/caffe/pull/1976
# shared im2col for reducing memory usage
hub merge https://github.com/BVLC/caffe/pull/2009
## commit

cat <<"END" > README.md
This is a pre-release Caffe branch for fully convolutional networks. This includes unmerged PRs and no guarantees.
Everything here is subject to change, including the history of this branch.
See `future.sh` for details.

END
git add README.md
git add future.sh
git commit -m 'add README + creation script'
